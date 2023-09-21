import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader

from upsell.configs import load_yaml_config
from upsell.event_seq_dataset import EventSeqDataset
from upsell.metric_tracker import MetricTracker
from upsell.rnn import ExplainableRecurrentPointProcess
from upsell.utils.data_loader import split_data_loader, convert_to_bucketed_data_loader
from upsell.utils.env import get_freer_gpu, set_rand_seed
from upsell.utils.s3 import load_event_type_names, load_numpy_data, load_pytorch_object,\
    save_pytorch_dataset, save_pytorch_model, save_attributions


def init_env():
    config = load_yaml_config('upsell/config.yml')
    set_rand_seed(config.env.seed, config.env.cuda)
    device = get_device(cuda=config.env.cuda)

    config.tenant_id = 1309
    config.run_date = '2023-07-01'
    config.bucket = 'ceasterwood'
    config.sampling = None

    event_type_names = load_event_type_names(config.bucket, config.tenant_id, config.run_date, config.sampling)
    config.model.n_event_types = event_type_names.shape[0]
    config.attribution.event_type_names = event_type_names['event_type'].to_list()
    return device, config


def run(device, configs, tune_params=True):
    # Get event sequences
    train_event_seqs = load_event_seqs(configs.tenant_id, configs.run_date, dataset='train')
    train_data_loader, valid_data_loader = init_data_loader(train_event_seqs, configs.data_loader, dataset='train')

    # Train model
    if tune_params:
        model = train_with_tuning(device, configs)
    else:
        untrained_model = init_model(device, configs.model)
        model = train(train_data_loader, valid_data_loader, untrained_model, device, configs.train,
                      configs.tenant_id, configs.run_date, tune_params=False)
        save_pytorch_model(model, 'ceasterwood', configs.tenant_id, configs.run_date, None)

    # Evaluate training history
    history = pd.read_csv(f'{configs.tenant_id}/{configs.run_date}/history.csv')
    plot_training_loss(history, tune_metric=configs.train.tune_metric,
                       filepath=f'{configs.tenant_id}/{configs.run_date}/training_loss.png')

    # Evaluate model on test set
    test_event_seqs = load_event_seqs(configs.tenant_id, configs.run_date, dataset='test')
    metrics = calculate_test_metrics(test_event_seqs, model, device, loader_configs=configs.data_loader)
    pd.DataFrame.from_dict({k: v.avg for k, v in metrics.items()}, orient='index')\
        .to_json(f'{configs.tenant_id}/{configs.run_date}/test_metrics.json')
    plot_avg_incidence_at_k(metrics, ks=model.ks,
                            filepath=f'{configs.tenant_id}/{configs.run_date}/avg_incidence_at_k.png')

    # Calculate infectivity
    if not configs.skip_eval_infectivity:
        attribution_matrix = calculate_infectivity(
            train_event_seqs,
            model,
            device,
            loader_configs=configs.data_loader,
            attribution_configs=configs.attribution
        )
        save_attributions(attribution_matrix, configs.bucket, configs.tenant_id, configs.run_date)
        print(attribution_matrix.shape)

    # Predict future event intensities on test set
    model = load_pytorch_object(configs.bucket, configs.tenant_id, configs.run_date, configs.sampling, 'model')
    pred_event_seqs = load_event_seqs(configs.tenant_id, configs.run_date, dataset='pred')
    intensities, cumulants = predict(
        pred_event_seqs,
        model,
        device,
        loader_configs=configs.data_loader,
        time_steps=range(configs.predict.min_time_steps, configs.predict.max_time_steps + 1)
    )
    for dataset, name in [(intensities, 'event_intensities'), (cumulants, 'event_cumulants')]:
        print(name, dataset.shape)
        save_pytorch_dataset(dataset, configs.bucket, configs.tenant_id,
                             configs.run_date, configs.sampling, name)


def get_device(cuda: bool, dynamic: bool = False):
    if torch.cuda.is_available() and cuda:
        if dynamic:
            device = torch.device('cuda', get_freer_gpu(by='n_proc'))
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def load_event_seqs(tenant_id, run_date, dataset='train'):
    bucket = 'ceasterwood'
    sampling = None
    data = load_numpy_data(bucket, tenant_id, run_date, sampling, dataset)
    event_seqs = data['event_seqs']

    if dataset == 'test':
        # Sort test_event_seqs by sequence length
        event_seqs = sorted(event_seqs, key=lambda seq: seq.shape[0])

    return event_seqs


def init_data_loader(event_seqs, loader_configs, dataset: str = 'train', attribution: bool = False):
    if not attribution and type(loader_configs['batch_size']) != int:
        # The train configs are from config.yml and not for tuning
        loader_configs['batch_size'] = loader_configs['batch_size']['default']

    data_loader_args = {
        'batch_size': loader_configs['attr_batch_size'] if attribution else loader_configs['batch_size'],
        'collate_fn': EventSeqDataset.collate_fn,
        'num_workers': loader_configs['num_workers'],
    }
    shuffle = (dataset == 'train') and not attribution

    data_loader = DataLoader(
        EventSeqDataset(event_seqs), shuffle=shuffle, **data_loader_args
    )

    if attribution:
        bucketed_data_loader = convert_to_bucketed_data_loader(
            data_loader, keys=[x.shape[0] for x in data_loader.dataset]
        )
        return bucketed_data_loader
    elif dataset == 'train':
        train_data_loader, valid_data_loader = split_data_loader(
            data_loader, loader_configs['train_validation_split']
        )
        if loader_configs['bucket_seqs']:
            train_data_loader = convert_to_bucketed_data_loader(
                train_data_loader, keys=[x.shape[0] for x in train_data_loader.dataset]
            )
        valid_data_loader = convert_to_bucketed_data_loader(
            valid_data_loader, keys=[x.shape[0] for x in valid_data_loader.dataset], shuffle_same_key=False
        )
        return train_data_loader, valid_data_loader
    else:
        return data_loader


def init_model(device, model_configs):
    if type(model_configs['embedding_dim']) != int:
        # The model configs are from config.yml and not for tuning
        model_configs['embedding_dim'] = model_configs['embedding_dim']['default']
        model_configs['hidden_size'] = model_configs['hidden_size']['default']
        model_configs['dropout'] = model_configs['dropout']['default']

    model = ExplainableRecurrentPointProcess(
        n_event_types=model_configs['n_event_types'],
        embedding_dim=model_configs['embedding_dim'],
        hidden_size=model_configs['hidden_size'],
        rnn=model_configs['rnn'],
        dropout=model_configs['dropout'],
        basis_type=model_configs['basis_type'],
        basis_means=model_configs['basis_means'],
        max_log_basis_weight=model_configs['max_log_basis_weight'],
        ks=model_configs['ks'],
    )
    model = model.to(device)
    return model


def train(train_data_loader, valid_data_loader, model, device, train_configs, tenant_id, run_date, tune_params=True):
    bucket = 'ceasterwood'
    sampling = None

    if type(train_configs['lr']) != float:
        # The train configs are from config.yml and not for tuning
        train_configs['lr'] = train_configs['lr']['default']

    optimizer = getattr(torch.optim, train_configs['optimizer'])(
        model.parameters(), lr=train_configs['lr']
    )

    model.train()

    best_metric = float('inf')
    best_epoch = 0

    epoch_metrics = {
        loss_type: {metric: MetricTracker(metric=metric) for metric in model.metrics}
        for loss_type in ['train', 'valid']
    }
    dt = []
    for epoch in range(train_configs['epochs']):
        start = datetime.now()
        print(f'Epoch {epoch}: {start}')
        train_metrics, valid_metrics = model.train_epoch(
            train_data_loader,
            optimizer,
            valid_data_loader,
            device=device,
            l2_reg=train_configs['l2_reg'],
        )
        # Store training and validation metrics for this epoch
        for loss_type in ['train', 'valid']:
            for metric in model.metrics:
                epoch_metrics[loss_type][metric].update(
                    eval(f'{loss_type}_metrics')[metric].avg,
                    n=eval(f'{loss_type}_metrics')[metric].count,
                )
        end = datetime.now()
        dt.append((end - start).total_seconds())

        tune_metric = train_configs['tune_metric']
        msg = f'[Training] Epoch={epoch} {tune_metric}={train_metrics[tune_metric].avg:.4f}'
        print(msg)  # logger.info(msg)
        msg = f'[Validation] Epoch={epoch} {tune_metric}={valid_metrics[tune_metric].avg:.4f}'
        print(msg)  # logger.info(msg)

        if valid_metrics[tune_metric].avg < best_metric:
            best_epoch = epoch
            best_metric = valid_metrics[tune_metric].avg
            if not tune_params:
                save_pytorch_model(model, bucket, tenant_id, run_date, sampling)

        if epoch - best_epoch >= train_configs['patience']:
            print(f'Stopped training early at epoch {epoch}: ' +
                  f'Failed to improve validation {tune_metric} in last {train_configs["patience"]} epochs')
            break

    if not tune_params:
        # Reset model to the last-saved (best) version of the model
        model = load_pytorch_object(bucket, tenant_id, run_date, sampling, 'model')

    # Save training history
    history = pd.DataFrame({
        'epoch': range(epoch + 1),
        'dt': dt,
    })
    for loss_type in ['train', 'valid']:
        for metric in model.metrics:
            history[f'{loss_type}_{metric}'] = epoch_metrics[loss_type][metric].values

    if tune_params:
        checkpoint_data = {
            'epoch': epoch,
            'history': history,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {k: v.avg for k, v in epoch_metrics['valid'].items()},
            checkpoint=checkpoint,
        )
    else:
        history.to_csv(f'{tenant_id}/{run_date}/history.csv', index=False)

    return model


def train_with_tuning(device, configs):
    def __train_with_tuning(__search_space, tenant_id, run_date):
        train_event_seqs = load_event_seqs(tenant_id, run_date, dataset='train')
        train_data_loader, valid_data_loader = init_data_loader(
            train_event_seqs, __search_space['data_loader'], dataset='train')
        untrained_model = init_model(device, __search_space['model'])
        _ = train(train_data_loader, valid_data_loader, untrained_model, device, __search_space['train'],
                  tenant_id, run_date, tune_params=True)

    # Create search space for hyperparameters
    search_space = {
        'data_loader': {
            'train_validation_split': configs.data_loader.train_validation_split,
            'bucket_seqs': configs.data_loader.bucket_seqs,
            'batch_size': tune.choice([2 ** i for i in range(
                configs.data_loader.batch_size.min_pow_2, configs.data_loader.batch_size.max_pow_2)]),
            'num_workers': configs.data_loader.num_workers,
        },
        'model': {
            'n_event_types': configs.model.n_event_types,
            'embedding_dim': tune.choice([2 ** i for i in range(
                configs.model.embedding_dim.min_pow_2, configs.model.embedding_dim.max_pow_2)]),
            'hidden_size': tune.choice([2 ** i for i in range(
                configs.model.hidden_size.min_pow_2, configs.model.hidden_size.max_pow_2)]),
            'rnn': configs.model.rnn,
            'dropout': tune.quniform(configs.model.dropout.min, configs.model.dropout.max, 0.01),
            'basis_type': configs.model.basis_type,
            'basis_means': configs.model.basis_means,
            'max_log_basis_weight': configs.model.max_log_basis_weight,
            'ks': configs.model.ks,
        },
        'train': {
            'optimizer': configs.train.optimizer,
            'lr': tune.qloguniform(configs.train.lr.min, configs.train.lr.max, 1.e-5),
            'epochs': configs.train.epochs,
            'l2_reg': configs.train.l2_reg,
            'tune_metric': configs.train.tune_metric,
            'patience': configs.train.patience,
        }
    }

    # Test different hyperparameter combinations using AsyncHyperBand algorithm
    scheduler = ASHAScheduler(
        metric=configs.train.tune_metric,
        mode='min',
        max_t=10,
        grace_period=configs.tuning.grace_period,
        reduction_factor=configs.tuning.reduction_factor,
    )
    result = tune.run(
        partial(__train_with_tuning, tenant_id=configs.tenant_id, run_date=configs.run_date),
        resources_per_trial={'cpu': 2, 'gpu': 0},
        config=search_space,
        num_samples=configs.tuning.n_param_combos,
        scheduler=scheduler,
    )

    # Initialize model with best hyperparameters
    best_trial = result.get_best_trial(configs.train.tune_metric, 'min', 'last')
    best_model = init_model(device, best_trial.config)

    # Load weights from best trial
    best_checkpoint_data = best_trial.checkpoint.to_air_checkpoint().to_dict()
    best_model.load_state_dict(best_checkpoint_data['model_state_dict'])
    return best_model


def calculate_test_metrics(event_seqs, model, device, loader_configs):
    data_loader = init_data_loader(event_seqs, loader_configs, dataset='test')
    metrics = model.evaluate(data_loader, device=device)
    msg = '[Test] ' + ', '.join(f'{k}={v.avg:.4f}' for k, v in metrics.items())
    print(msg)
    return metrics


def calculate_infectivity(event_seqs, model, device, loader_configs, attribution_configs):
    attr_data_loader = init_data_loader(event_seqs, loader_configs, attribution=True)
    attr_matrix = model.get_infectivity(
        attr_data_loader,
        device=device,
        steps=attribution_configs.steps,
        occurred_type_only=attribution_configs.occurred_type_only
    )
    attr_df = pd.DataFrame(attr_matrix.numpy(), columns=attribution_configs.event_type_names,
                           index=attribution_configs.event_type_names)
    return attr_df


def plot_training_loss(history, tune_metric, filepath=None):
    plt.plot(history[f'train_{tune_metric}'], label='train')
    plt.plot(history[f'valid_{tune_metric}'], label='valid')
    plt.xlabel('epoch')
    plt.ylabel(tune_metric)
    plt.legend()
    if filepath:
        plt.savefig(filepath)
    plt.show()


def plot_avg_incidence_at_k(metrics, ks, filepath=None):
    incidences = [metrics[f'avg_incidence_at_{k}'].avg for k in ks]
    plt.plot(ks, incidences)
    plt.plot([0, max(ks)], [0, max(ks)], linestyle=':', color='black')
    plt.title('Calibration Plot')
    plt.xlabel('Predicted Incidence')
    plt.ylabel('True Average Incidence')
    if filepath:
        plt.savefig(filepath)
    plt.show()


def predict(event_seqs, model, device, loader_configs, time_steps=range(1, 5)):
    data_loader = init_data_loader(event_seqs, loader_configs, dataset='pred')
    intensities, cumulants = \
        model.predict_future_event_intensities(data_loader, device, time_steps)
    return intensities, cumulants


if __name__ == '__main__':
    DEVICE, CONFIG = init_env()
    run(DEVICE, CONFIG, tune_params=True)

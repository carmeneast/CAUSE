import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from ray import tune
from ray.air import session
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Union

from upsell.configs import DotDict, load_yaml_config
from upsell.event_seq_dataset import EventSeqDataset
from upsell.metric_tracker import MetricTracker
from upsell.rnn import ExplainableRecurrentPointProcess
from upsell.utils.data_loader import split_data_loader, convert_to_bucketed_data_loader
from upsell.utils.env import get_freer_gpu, set_rand_seed
from upsell.utils.s3 import load_event_type_names, load_numpy_data, load_pytorch_object, \
    save_pytorch_dataset, save_pytorch_model, save_attributions


def init_env(tenant_id: int, model_id: str, run_date: str, bucket: str = 'ceasterwood') -> Tuple[torch.device, DotDict]:
    config = load_yaml_config('upsell/config.yml')
    set_rand_seed(config.env.seed, config.env.cuda)
    device = get_device(cuda=config.env.cuda)

    config.tenant_id = tenant_id
    config.model_id = model_id
    config.run_date = run_date
    config.bucket = bucket

    config.filepath = f'{config.tenant_id}/{config.model_id}/{config.run_date}'

    event_type_names = load_event_type_names(config.bucket, config.tenant_id, config.model_id, config.run_date)
    config.model.n_event_types = event_type_names.shape[0]
    config.attribution.event_type_names = event_type_names['event_type'].to_list()
    return device, config


def run(device: torch.device, config: DotDict, tune_params: bool = True) -> None:
    # Get event sequences
    train_event_seqs = load_event_seqs(config.tenant_id, config.model_id, config.run_date, dataset='train')

    # Train model
    if tune_params:
        model = train_with_tuning(device, config)
    else:
        train_data_loader, valid_data_loader = init_data_loader(train_event_seqs, config.data_loader, dataset='train')
        untrained_model = init_model(config.model, device=device)
        model = train(train_data_loader, valid_data_loader, untrained_model, config.train,
                      config.tenant_id, config.model_id, config.run_date, config.filepath,
                      device=device, tune_params=False)
        save_pytorch_model(model, config.bucket, config.tenant_id, config.run_date, config.model_id)

    # Evaluate training history
    history = pd.read_csv(f'{config.filepath}/history.csv')
    plot_training_loss(history, tune_metric=config.train.tune_metric, filepath=f'{config.filepath}/training_loss.png')

    # Evaluate model on test set
    test_event_seqs = load_event_seqs(config.tenant_id, config.model_id, config.run_date, dataset='test')
    metrics = calculate_test_metrics(test_event_seqs, model, config.attribution.event_type_names, device,
                                     loader_configs=config.data_loader)
    pd.DataFrame.from_dict(metrics, orient='index').to_json(f'{config.filepath}/test_metrics.json')
    plot_avg_incidence_at_k(metrics['avg_incidences'], filepath=f'{config.filepath}/avg_incidence_at_k.png')
    plot_avg_incidence_at_k(metrics['avg_incidences_nb_opp'], event_type='Opened New Business Opportunity',
                            filepath=f'{config.filepath}/avg_incidence_at_k_nb_opp.png')
    plot_avg_incidence_at_k(metrics['avg_incidences_pc_opp'], event_type='Opened Post-Customer Opportunity',
                            filepath=f'{config.filepath}/avg_incidence_at_k_pc_opp.png')

    # Calculate infectivity
    if not config.attribution.skip_eval_infectivity:
        attribution_matrix = calculate_infectivity(
            train_event_seqs,
            model,
            device,
            loader_configs=config.data_loader,
            attribution_configs=config.attribution
        )
        save_attributions(attribution_matrix, config.bucket, config.tenant_id, config.run_date, config.model_id)
        print(attribution_matrix.shape)

    # Predict future event intensities on test set
    pred_event_seqs = load_event_seqs(config.tenant_id, config.model_id, config.run_date, dataset='pred')
    intensities, cumulants = predict(
        pred_event_seqs,
        model,
        device,
        loader_configs=config.data_loader,
        time_steps=range(config.predict.min_time_steps, config.predict.max_time_steps + 1)
    )
    for dataset, name in [(intensities, 'event_intensities'), (cumulants, 'event_cumulants')]:
        print(name, dataset.shape)
        save_pytorch_dataset(dataset, config.bucket, config.tenant_id,
                             config.run_date, config.model_id, name)


def get_device(cuda: bool, dynamic: bool = False) -> torch.device:
    if torch.cuda.is_available() and cuda:
        if dynamic:
            device = torch.device('cuda', get_freer_gpu(by='n_proc'))
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def load_event_seqs(tenant_id: int, model_id: str, run_date: str, dataset: str = 'train') -> EventSeqDataset:
    bucket = 'ceasterwood'
    data = load_numpy_data(bucket, tenant_id, run_date, model_id, dataset)
    event_seqs = data['event_seqs']

    if dataset == 'test':
        # Sort test_event_seqs by sequence length
        event_seqs = sorted(event_seqs, key=lambda seq: seq.shape[0])

    return event_seqs


def init_data_loader(event_seqs: EventSeqDataset, loader_configs: DotDict, dataset: str = 'train',
                     attribution: bool = False) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    if attribution:
        batch_size = loader_configs.attr_batch_size
    elif isinstance(loader_configs['batch_size'], int):
        batch_size = loader_configs['batch_size']
    else:
        batch_size = loader_configs.batch_size.default

    data_loader_args = {
        'batch_size': batch_size,
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


def init_model(model_configs: DotDict, device: Optional[torch.device] = None) -> ExplainableRecurrentPointProcess:
    model = ExplainableRecurrentPointProcess(
        n_event_types=model_configs['n_event_types'],
        embedding_dim=model_configs['embedding_dim'] if isinstance(model_configs['embedding_dim'], int)
        else model_configs.embedding_dim.default,
        hidden_size=model_configs['hidden_size'] if isinstance(model_configs['hidden_size'], int)
        else model_configs.hidden_size.default,
        rnn=model_configs['rnn'],
        dropout=model_configs['dropout'] if isinstance(model_configs['dropout'], float)
        else model_configs.dropout.default,
        basis_type=model_configs['basis_type'],
        basis_means=model_configs['basis_means'],
        max_log_basis_weight=model_configs['max_log_basis_weight'],
    )
    if device is not None:
        model = model.to(device)
    return model


def train(train_data_loader: DataLoader, valid_data_loader: DataLoader,
          model: ExplainableRecurrentPointProcess, train_configs: DotDict,
          tenant_id: int, model_id: str, run_date: str, filepath: str,
          device: Optional[torch.device] = None, tune_params: bool = True) -> ExplainableRecurrentPointProcess:
    bucket = 'ceasterwood'
    patience = train_configs['patience'] if tune_params else train_configs.patience.no_tuning

    if device is None:
        device = get_device(cuda=True)
        model.to(device)

    optimizer = getattr(torch.optim, train_configs['optimizer'])(
        model.parameters(), lr=train_configs['lr'] if isinstance(train_configs['lr'], float)
        else train_configs.lr.default
    )

    # Load existing checkpoint if it exists
    # TODO: does this exist if not doing tuning?
    checkpoint = session.get_checkpoint() if tune_params else None
    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state['epoch']
        best_epoch = checkpoint_state['epoch']
        best_metric = checkpoint_state['history'][f'valid_{train_configs["tune_metric"]}'].to_list()[-1]
        history = checkpoint_state['history']
        model.load_state_dict(checkpoint_state['model_state_dict'])
        optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
    else:
        start_epoch = 0
        best_epoch = 0
        best_metric = float('inf')
        history = pd.DataFrame()

    model.train()

    metrics = ['nll']
    epoch_metrics = {
        loss_type: {metric: MetricTracker(metric=metric) for metric in metrics}
        for loss_type in ['train', 'valid']
    }
    for epoch in range(start_epoch, train_configs['epochs']):
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
            for metric in metrics:
                n = eval(f'{loss_type}_metrics')[metric].count
                val = eval(f'{loss_type}_metrics')[metric].avg
                if isinstance(val, torch.Tensor):
                    val = val.item()
                epoch_metrics[loss_type][metric].update(val, n=n)
        end = datetime.now()

        # Add epoch metrics to history
        history = pd.concat([
            history,
            pd.DataFrame({
                'epoch': [epoch],
                'dt': [(end - start).total_seconds()],
                **{f'{loss_type}_{metric}': [epoch_metrics[loss_type][metric].avg]
                   for loss_type in ['train', 'valid'] for metric in metrics}
            })
        ]).reset_index(drop=True)

        tune_metric = train_configs['tune_metric']
        msg = f'[Training] Epoch={epoch} {tune_metric}={train_metrics[tune_metric].avg:.4f}'
        print(msg)  # logger.info(msg)
        msg = f'[Validation] Epoch={epoch} {tune_metric}={valid_metrics[tune_metric].avg:.4f}'
        print(msg)  # logger.info(msg)

        if tune_params:
            # Report metrics to Ray Tune. The scheduler will determine if training should stop early.
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

        # Manually check if training should stop early
        # Note: Even if using Ray Tune, we want to manually cut off training if the error is not improving
        # Ray Tune only stops training at specific checkpoints, like epoch = 1, 2, 8, 16, 64, etc.
        if valid_metrics[tune_metric].avg < best_metric:
            best_epoch = epoch
            best_metric = valid_metrics[tune_metric].avg
            if not tune_params:
                save_pytorch_model(model, bucket, tenant_id, run_date, model_id)

        if epoch - best_epoch >= patience:
            print(f'Stopped training early at epoch {epoch}: ' +
                  f'Failed to improve validation {tune_metric} in last {patience} epochs')
            break

    if not tune_params:
        # Reset model to the last-saved (best) version of the model
        model = load_pytorch_object(bucket, tenant_id, run_date, model_id, 'model')

        # Save training history
        history.to_csv(f'{filepath}/history.csv', index=False)

    return model


def train_with_tuning(device: torch.device, config: DotDict) -> ExplainableRecurrentPointProcess:
    def __train_with_tuning(__search_space, tenant_id, run_date, model_id, filepath):
        train_event_seqs = load_event_seqs(tenant_id, model_id, run_date, dataset='train')
        train_data_loader, valid_data_loader = init_data_loader(
            train_event_seqs, __search_space['data_loader'], dataset='train')
        untrained_model = init_model(__search_space['model'])
        _ = train(train_data_loader, valid_data_loader, untrained_model, __search_space['train'],
                  tenant_id, model_id, run_date, filepath, tune_params=True)

    # Create search space for hyperparameters
    search_space = {
        'data_loader': {
            'train_validation_split': config.data_loader.train_validation_split,
            'bucket_seqs': config.data_loader.bucket_seqs,
            'batch_size': tune.choice([2 ** i for i in range(
                config.data_loader.batch_size.min_pow_2, config.data_loader.batch_size.max_pow_2)]),
            'num_workers': config.data_loader.num_workers,
        },
        'model': {
            'n_event_types': config.model.n_event_types,
            'embedding_dim': tune.choice([2 ** i for i in range(
                config.model.embedding_dim.min_pow_2, config.model.embedding_dim.max_pow_2)]),
            'hidden_size': tune.choice([2 ** i for i in range(
                config.model.hidden_size.min_pow_2, config.model.hidden_size.max_pow_2)]),
            'rnn': config.model.rnn,
            'dropout': tune.quniform(config.model.dropout.min, config.model.dropout.max, 0.01),
            'basis_type': config.model.basis_type,
            'basis_means': config.model.basis_means,
            'max_log_basis_weight': config.model.max_log_basis_weight,
        },
        'train': {
            'optimizer': config.train.optimizer,
            'lr': tune.qloguniform(config.train.lr.min, config.train.lr.max, 1.e-5),
            'epochs': config.train.epochs,
            'l2_reg': config.train.l2_reg,
            'tune_metric': config.train.tune_metric,
            'patience': config.train.patience.ray_tune,
        }
    }

    # Test different hyperparameter combinations using AsyncHyperBand algorithm
    scheduler = ASHAScheduler(
        metric=config.train.tune_metric,
        mode='min',
        max_t=config.tuning.max_training_iterations,
        grace_period=config.tuning.grace_period,
        reduction_factor=config.tuning.reduction_factor,
    )
    result = tune.run(
        partial(__train_with_tuning, tenant_id=config.tenant_id, run_date=config.run_date,
                model_id=config.model_id, filepath=config.filepath),
        # resources_per_trial={'cpu': 8, 'gpu': 0},
        config=search_space,
        num_samples=config.tuning.n_param_combos,
        scheduler=scheduler,
    )

    # Initialize model with best hyperparameters
    tune_metric = config.train.tune_metric
    best_trial = result.get_best_trial(tune_metric, 'min', 'last')
    print(f'Best trial config: {best_trial.config}')
    print(f'Best trial validation {tune_metric}: {best_trial.last_result[tune_metric]}')
    best_model = init_model(best_trial.config['model'], device)

    # Load weights from best trial
    best_checkpoint_data = best_trial.checkpoint.to_air_checkpoint().to_dict()
    best_model.load_state_dict(best_checkpoint_data['model_state_dict'])

    # Save results from best trial
    save_pytorch_model(best_model, config.bucket, config.tenant_id, config.run_date, config.model_id)
    best_checkpoint_data['history'].to_csv(f'{config.tenant_id}/{config.run_date}/history.csv', index=False)

    return best_model


def calculate_test_metrics(event_seqs: EventSeqDataset, model: ExplainableRecurrentPointProcess,
                           event_type_names: list[str], device: torch.device,
                           loader_configs: DotDict) -> Dict[str, Union[float, Dict[float, float]]]:
    data_loader = init_data_loader(event_seqs, loader_configs, dataset='test')
    ks = np.arange(0.0, 2.1, 0.01).tolist()

    # Metrics for all events
    metrics = model.evaluate(data_loader, device=device)
    metrics = {k: v.avg for k, v in metrics.items()}
    print(metrics)
    avg_incidences = model.generate_calibration_curve(data_loader, ks, device=device)

    # Metrics for specific event types
    nb_idx = np.where(np.array(event_type_names) == 'opened_new_business_opportunity')[0][0]
    # nb_metrics = model.evaluate(data_loader, event_index=nb_idx, device=device)
    # print(nb_metrics)
    avg_incidences_nb_opp = model.generate_calibration_curve(data_loader, ks, event_index=nb_idx, device=device)

    pc_idx = np.where(np.array(event_type_names) == 'opened_post_customer_opportunity')[0][0]
    # pc_metrics = model.evaluate(data_loader, event_index=pc_idx, device=device)
    # print(pc_metrics)
    avg_incidences_pc_opp = model.generate_calibration_curve(data_loader, ks, event_index=pc_idx, device=device)
    return {
        **{k: v.avg for k, v in metrics.items()},
        # **{f'{k}_nb_opp': v.avg for k, v in nb_metrics.items()},
        # **{f'{k}_pc_opp': v.avg for k, v in pc_metrics.items()},
        'avg_incidences': {k: avg_incidences[k].avg for k in ks},
        'avg_incidences_nb_opp': {k: avg_incidences_nb_opp[k].avg for k in ks},
        'avg_incidences_pc_opp': {k: avg_incidences_pc_opp[k].avg for k in ks},
    }


def calculate_infectivity(event_seqs: EventSeqDataset, model: ExplainableRecurrentPointProcess, device: torch.device,
                          loader_configs: DotDict, attribution_configs: DotDict) -> pd.DataFrame:
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


def plot_training_loss(history: pd.DataFrame, tune_metric: str, filepath: Optional[str] = None) -> None:
    plt.plot(history[f'train_{tune_metric}'], label='train')
    plt.plot(history[f'valid_{tune_metric}'], label='valid')
    plt.xlabel('epoch')
    plt.ylabel(tune_metric)
    plt.legend()
    if filepath:
        plt.savefig(filepath)
    plt.show()


def plot_avg_incidence_at_k(metrics: dict[float, float], event_type: Optional[str] = None,
                            filepath: Optional[str] = None) -> None:
    ks = list(metrics.keys())
    incidences = list(metrics.values())
    plt.plot(ks, incidences)
    plt.plot([0, max(ks)], [0, max(ks)], linestyle=':', color='black')
    if event_type:
        plt.title(f'Calibration Plot ({event_type})')
    else:
        plt.title('Calibration Plot (All Events)')
    plt.xlabel('Predicted Incidence')
    plt.ylabel('True Average Incidence')
    if filepath:
        plt.savefig(filepath)
    plt.show()


def predict(event_seqs: EventSeqDataset, model: ExplainableRecurrentPointProcess, device: torch.device,
            loader_configs: DotDict, time_steps: list[int] = range(1, 5)):
    data_loader = init_data_loader(event_seqs, loader_configs, dataset='pred')
    intensities, cumulants = \
        model.predict_future_event_intensities(data_loader, device, time_steps)
    return intensities, cumulants


if __name__ == '__main__':
    TENANT_ID = 1309
    RUN_DATE = '2023-11-01'
    BUCKET = 'ceasterwood'
    MODEL_ID = '2'
    TUNE_PARAMS = True

    DEVICE, CONFIG = init_env(TENANT_ID, MODEL_ID, RUN_DATE, BUCKET)
    run(DEVICE, CONFIG, TUNE_PARAMS)

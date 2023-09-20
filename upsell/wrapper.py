import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
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
from upsell.utils.s3 import load_numpy_data, load_pytorch_object,\
    save_pytorch_dataset, save_pytorch_model, save_attributions


class CauseWrapper:

    def __init__(self, tenant_id, run_date, sampling=None, bucket='ceasterwood'):
        self.tenant_id = tenant_id
        self.run_date = run_date
        self.sampling = sampling
        self.bucket = bucket

        self.CONFIG = load_yaml_config('upsell/config.yml')

        # initialization
        set_rand_seed(self.CONFIG.env.seed, self.CONFIG.env.cuda)
        self.device = self.get_device()
        # init_logging(output_path)
        # logger = get_logger(__file__)

        # initialize model artifacts
        self.n_event_types, self.event_type_names = None, None
        self.model = None

    def run(self, tune_params=True):
        # Get event sequences
        train_event_seqs = self.load_event_seqs(dataset='train')
        test_event_seqs = self.load_event_seqs(dataset='test')
        train_data_loader, valid_data_loader = self.init_data_loader(train_event_seqs, dataset='train')

        # Train model
        if tune_params:
            self.train_with_tuning(train_data_loader, valid_data_loader)
        else:
            self.init_model()
            self.train(train_data_loader, valid_data_loader, tune_params=False)
        history = pd.read_csv(f'{self.tenant_id}/{self.run_date}/history.csv')
        self.plot_training_loss(history, self.CONFIG.train.tune_metric)

        # Evaluate model on test set
        metrics = self.calculate_test_metrics(test_event_seqs)
        self.plot_avg_incidence_at_k(metrics)

        # Calculate infectivity
        attribution_matrix = self.calculate_infectivity(train_event_seqs)
        if attribution_matrix is not None:
            print(attribution_matrix.shape)

        # Predict future event intensities on test set
        pred_event_seqs = self.load_event_seqs(dataset='pred')
        intensities, cumulants, log_basis_weights = self.predict(
            pred_event_seqs,
            time_steps=range(self.CONFIG.predict.min_time_steps, self.CONFIG.predict.max_time_steps + 1)
        )
        print(intensities.shape, cumulants.shape, log_basis_weights.shape)

    def get_device(self, dynamic=False):
        if torch.cuda.is_available() and self.CONFIG.env.cuda:
            if dynamic:
                device = torch.device('cuda', get_freer_gpu(by='n_proc'))
            else:
                device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return device

    def load_event_seqs(self, dataset='train'):
        data = load_numpy_data(self.bucket, self.tenant_id, self.run_date, self.sampling, dataset)
        self.n_event_types = data['n_event_types']
        self.event_type_names = data['event_type_names']
        event_seqs = data['event_seqs']

        if dataset == 'test':
            # Sort test_event_seqs by sequence length
            event_seqs = sorted(event_seqs, key=lambda seq: seq.shape[0])

        return event_seqs

    def init_data_loader(self, event_seqs, dataset: str = 'train', attribution: bool = False):
        configs = self.CONFIG.data_loader
        data_loader_args = {
            'batch_size': configs.attr_batch_size if attribution else configs.batch_size,
            'collate_fn': EventSeqDataset.collate_fn,
            'num_workers': configs.num_workers,
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
                data_loader, configs.train_validation_split
            )
            if configs.bucket_seqs:
                train_data_loader = convert_to_bucketed_data_loader(
                    train_data_loader, keys=[x.shape[0] for x in train_data_loader.dataset]
                )
            valid_data_loader = convert_to_bucketed_data_loader(
                valid_data_loader, keys=[x.shape[0] for x in valid_data_loader.dataset], shuffle_same_key=False
            )
            return train_data_loader, valid_data_loader
        else:
            return data_loader

    def init_model(self, param_space=None):
        """
        Initialize ExplainableRecurrentPointProcess model object
        :param param_space: (Optional) ray-tune search space for hyperparameters
            If not provided, use default hyperparameters
        :return: None
        """
        config = self.CONFIG.model
        if param_space is None:
            param_space = {
                'embedding_dim': config.embedding_dim.default,
                'hidden_size': config.hidden_size.default,
                'dropout': config.dropout.default,
            }

        self.model = ExplainableRecurrentPointProcess(
            n_event_types=self.n_event_types,
            embedding_dim=param_space['embedding_dim'],
            hidden_size=param_space['hidden_size'],
            rnn=config.rnn,
            dropout=param_space['dropout'],
            basis_type=config.basis_type,
            basis_means=config.basis_means,
            max_log_basis_weight=config.max_log_basis_weight,
            ks=config.ks,
        )
        self.model = self.model.to(self.device)

    def train(self, train_data_loader, valid_data_loader, tune_params=True):
        configs = self.CONFIG.train
        optimizer = getattr(torch.optim, configs.optimizer)(
            self.model.parameters(), lr=configs.lr
        )

        self.model.train()

        best_metric = float('inf')
        best_epoch = 0

        epoch_metrics = {
            loss_type: {metric: MetricTracker(metric=metric) for metric in self.model.metrics}
            for loss_type in ['train', 'valid']
        }
        dt = []
        for epoch in range(configs.epochs):
            start = datetime.now()
            print(f'Epoch {epoch}: {start}')
            train_metrics, valid_metrics = self.model.train_epoch(
                train_data_loader,
                optimizer,
                valid_data_loader,
                device=self.device,
                # Unload remaining configs besides optimizer
                **{k: v for k, v in configs.items() if k != 'optimizer'},
            )
            # Store training and validation metrics for this epoch
            for loss_type in ['train', 'valid']:
                for metric in self.model.metrics:
                    epoch_metrics[loss_type][metric].update(
                        eval(f'{loss_type}_metrics')[metric].avg.item(),
                        n=eval(f'{loss_type}_metrics')[metric].count,
                    )
            end = datetime.now()
            dt.append((end - start).total_seconds())

            msg = f'[Training] Epoch={epoch} {configs.tune_metric}={train_metrics[configs.tune_metric].avg:.4f}'
            print(msg)  # logger.info(msg)
            msg = f'[Validation] Epoch={epoch} {configs.tune_metric}={valid_metrics[configs.tune_metric].avg:.4f}'
            print(msg)  # logger.info(msg)

            if valid_metrics[configs.tune_metric].avg < best_metric:
                best_epoch = epoch
                best_metric = valid_metrics[configs.tune_metric].avg
                save_pytorch_model(self.model, self.bucket, self.tenant_id, self.run_date, self.sampling)

            if epoch - best_epoch >= configs.patience:
                print(f'Stopped training early at epoch {epoch}: ' +
                      f'Failed to improve validation {configs.tune_metric} in last {configs.patience} epochs')
                break

        # Reset model to the last-saved (best) version of the model
        self.model = load_pytorch_object(self.bucket, self.tenant_id, self.run_date, self.sampling, 'model')

        # Save training history
        history = pd.DataFrame({
            'epoch': range(epoch + 1),
            'dt': dt,
        })
        for loss_type in ['train', 'valid']:
            for metric in self.model.metrics:
                history[f'{loss_type}_{metric}'] = epoch_metrics[loss_type][metric].values
        if self.sampling:
            filename = f'{self.tenant_id}/{self.run_date}/{self.sampling}/history.csv'
        else:
            filename = f'{self.tenant_id}/{self.run_date}/history.csv'
        history.to_csv(filename, index=False)

        if tune_params:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            checkpoint = Checkpoint.from_dict(checkpoint_data)

            session.report(
                {k: v.avg for k, v in epoch_metrics['valid'].items()},
                checkpoint=checkpoint,
            )

        return history

    def train_with_tuning(self, train_data_loader, valid_data_loader):
        def __train_with_tuning(_config):
            self.init_model(param_space=_config)
            self.train(train_data_loader, valid_data_loader, tune_params=True)

        # Create search space for hyperparameters
        config = {
            'dropout': tune.quniform(self.CONFIG.model.dropout.min, self.CONFIG.model.dropout.max, 0.01),
            'embedding_dim': tune.choice([2 ** i for i in range(
                self.CONFIG.model.embedding_dim.min_pow_2, self.CONFIG.model.embedding_dim.max_pow_2)]),
            'hidden_size': tune.choice([2 ** i for i in range(
                self.CONFIG.model.hidden_size.min_pow_2, self.CONFIG.model.hidden_size.max_pow_2)]),
        }

        # Test different hyperparameter combinations using AsyncHyperBand algorithm
        scheduler = ASHAScheduler(
            metric=self.CONFIG.train.tune_metric,
            mode='min',
            max_t=10,
            grace_period=self.CONFIG.tuning.grace_period,
            reduction_factor=self.CONFIG.tuning.reduction_factor,
        )
        result = tune.run(
            __train_with_tuning,
            resources_per_trial={'cpu': 2, 'gpu': 0},
            config=config,
            num_samples=self.CONFIG.tuning.n_param_combos,
            scheduler=scheduler,
        )

        # Initialize model with best hyperparameters
        best_trial = result.get_best_trial(self.CONFIG.train.tune_metric, 'min', 'last')
        self.init_model(param_space=best_trial.config)

        # Load weights from best trial
        best_checkpoint_data = best_trial.checkpoint.to_air_checkpoint().to_dict()
        self.model.load_state_dict(best_checkpoint_data['model_state_dict'])

    def plot_training_loss(self, history, tune_metric):
        plt.plot(history[f'train_{tune_metric}'], label='train')
        plt.plot(history[f'valid_{tune_metric}'], label='valid')
        plt.title(self.tenant_id)
        plt.xlabel('epoch')
        plt.ylabel(tune_metric)
        plt.legend()
        if self.sampling:
            plt.savefig(f'{self.tenant_id}/{self.run_date}/{self.sampling}/training_loss.png')
        else:
            plt.savefig(f'{self.tenant_id}/{self.run_date}/training_loss.png')
        plt.show()

    def plot_avg_incidence_at_k(self, metrics):
        ks = self.CONFIG.model.ks
        incidences = [metrics[f'avg_incidence_at_{k}'].avg for k in ks]
        plt.plot(ks, incidences)
        plt.plot([0, max(ks)], [0, max(ks)], linestyle=':', color='black')
        plt.title(f'{self.tenant_id} Calibration Plot')
        plt.xlabel('Predicted Incidence')
        plt.ylabel('True Average Incidence')
        if self.sampling:
            plt.savefig(f'{self.tenant_id}/{self.run_date}/{self.sampling}/avg_incidence_at_k.png')
        else:
            plt.savefig(f'{self.tenant_id}/{self.run_date}/avg_incidence_at_k.png')
        plt.show()

    def calculate_test_metrics(self, event_seqs):
        data_loader = self.init_data_loader(event_seqs, dataset='test')
        metrics = self.model.evaluate(data_loader, device=self.device)
        msg = '[Test] ' + ', '.join(f'{k}={v.avg:.4f}' for k, v in metrics.items())
        print(msg)  # logger.info(msg)
        if self.sampling:
            filename = f'{self.tenant_id}/{self.run_date}/{self.sampling}/test_metrics.json'
        else:
            filename = f'{self.tenant_id}/{self.run_date}/test_metrics.json'
        pd.DataFrame.from_dict({k: v.avg for k, v in metrics.items()}, orient='index')\
            .to_json(filename)
        return metrics

    def predict(self, event_seqs, time_steps=range(1, 5)):
        if self.model is None:
            self.model = load_pytorch_object(self.bucket, self.tenant_id, self.run_date, self.sampling, 'model')

        data_loader = self.init_data_loader(event_seqs, dataset='pred')
        intensities, cumulants, log_basis_weights = \
            self.model.predict_future_event_intensities(data_loader, self.device, time_steps)

        for dataset, name in [(intensities, 'event_intensities'),
                              (cumulants, 'event_cumulants'),
                              (log_basis_weights, 'log_basis_weights')]:
            save_pytorch_dataset(dataset, self.bucket, self.tenant_id, self.run_date, self.sampling, name)

        return intensities, cumulants, log_basis_weights

    def evaluate_event_prediction(self):
        # Function to evaluate predictions for a specific event type
        pass

    def calculate_infectivity(self, event_seqs):
        configs = self.CONFIG.attribution
        if not configs.skip_eval_infectivity:
            attr_data_loader = self.init_data_loader(event_seqs, attribution=True)
            attr_matrix = self.model.get_infectivity(
                attr_data_loader,
                device=self.device,
                steps=configs.steps,
                occurred_type_only=configs.occurred_type_only
            )
            index = self.event_type_names['event_type'].to_list()
            attr_df = pd.DataFrame(attr_matrix.numpy(), columns=index, index=index)
            save_attributions(attr_df, self.bucket, self.tenant_id, self.run_date, self.sampling)
        else:
            print('Skipping calculate infectivity')
            attr_matrix = None
        return attr_matrix

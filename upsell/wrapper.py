import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader

from upsell.configs import load_yaml_config
from upsell.event_seq_dataset import EventSeqDataset
from upsell.metric_tracker import MetricTracker
from upsell.rnn import ExplainableRecurrentPointProcess
from upsell.utils.data_loader import split_data_loader, convert_to_bucketed_data_loader
from upsell.utils.env import get_freer_gpu, set_rand_seed
from upsell.utils.s3 import load_numpy_data, load_pytorch_object, save_pytorch_dataset, save_pytorch_model


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

    def run(self):
        # Get event sequences
        train_event_seqs = self.load_event_seqs(dataset='train')
        test_event_seqs = self.load_event_seqs(dataset='test')
        train_data_loader, valid_data_loader = self.init_data_loader(train_event_seqs, dataset='train')

        # Train model
        self.init_model()
        history = self.train(train_data_loader, valid_data_loader)
        self.plot_training_loss(history, self.CONFIG.train.tune_metric)

        # Evaluate model on test set
        metrics = self.calculate_test_metrics(test_event_seqs)
        self.plot_avg_incidence_at_k(metrics)

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

    def init_data_loader(self, event_seqs, dataset='train'):
        configs = self.CONFIG.data_loader
        data_loader_args = {
            'batch_size': configs.batch_size,
            'collate_fn': EventSeqDataset.collate_fn,
            'num_workers': configs.num_workers,
        }

        shuffle = (dataset == 'train')
        data_loader = DataLoader(
            EventSeqDataset(event_seqs), shuffle=shuffle, **data_loader_args
        )

        if dataset == 'train':
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

    def init_model(self):
        self.model = ExplainableRecurrentPointProcess(n_event_types=self.n_event_types, **{**self.CONFIG.model})
        self.model = self.model.to(self.device)

    def train(self, train_data_loader, valid_data_loader):
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
        return history

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
        plt.title(self.tenant_id)
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

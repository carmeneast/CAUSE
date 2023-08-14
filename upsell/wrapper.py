import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from upsell.configs import load_yaml_config
from upsell.event_seq_dataset import EventSeqDataset
from upsell.metric_tracker import MetricTracker
from upsell.rnn import ExplainableRecurrentPointProcess
from upsell.s3 import load_numpy_data, load_pytorch_object, save_pytorch_dataset, save_pytorch_model
from upsell.utils import convert_to_bucketed_dataloader, get_freer_gpu, set_rand_seed, split_dataloader


class CauseWrapper:

    def __init__(self, tenant_id, bucket='ceasterwood'):
        self.tenant_id = tenant_id
        self.bucket = bucket

        self.CONFIG = load_yaml_config('upsell/config.yml')

        # initialization
        set_rand_seed(self.CONFIG.env.seed, self.CONFIG.env.cuda)
        self.device = self.get_device()
        # init_logging(output_path)
        # logger = get_logger(__file__)

        # initialize model artifacts
        self.n_types, self.event_type_names = None, None
        self.model = None
        self.account_ids = None

        self.data_loader_args = {
            'batch_size': self.CONFIG.data_loader.batch_size,
            'collate_fn': EventSeqDataset.collate_fn,
            'num_workers': self.CONFIG.data_loader.num_workers,
        }

    def run(self):
        # Get event sequences
        train_event_seqs, test_event_seqs = self.load_event_seqs()
        train_data_loader, valid_data_loader = self.init_data_loaders(train_event_seqs)

        # Trains model
        self.init_model()
        history = self.train(train_data_loader, valid_data_loader)
        self.plot_training_loss(history, self.CONFIG.train.tune_metric)

        # Evaluates model on test set
        metrics = self.calculate_test_metrics(test_event_seqs)
        self.plot_precision_at_k(metrics)

        # Predicts future event intensities on test set
        days = range(self.CONFIG.predict.min_days, self.CONFIG.predict.max_days + 1)
        intensities, cumulants, log_basis_weights = self.predict(test_event_seqs, days=days)
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

    def load_event_seqs(self):
        data = load_numpy_data(self.bucket, self.tenant_id)
        self.n_types = data['n_types']
        self.event_type_names = data['event_type_names']
        self.account_ids = data['account_ids']
        event_seqs = data['event_seqs']
        train_test_splits = data['train_test_splits']  # Contains 5 possible train-test splits

        # Randomly select a train-test split
        split_id = np.random.randint(5)
        train_event_seqs = event_seqs[train_test_splits[split_id][0]]
        test_event_seqs = event_seqs[train_test_splits[split_id][1]]

        # Sort test_event_seqs by sequence length
        test_event_seqs = sorted(test_event_seqs, key=lambda seq: seq.shape[0])
        return train_event_seqs, test_event_seqs

    def init_data_loaders(self, event_seqs):
        train_data_loader = DataLoader(
            EventSeqDataset(event_seqs), **self.data_loader_args
        )
        train_data_loader, valid_data_loader = split_dataloader(
            train_data_loader, self.CONFIG.data_loader.train_validation_split
        )
        if self.CONFIG.data_loader.bucket_seqs:
            train_data_loader = convert_to_bucketed_dataloader(
                train_data_loader, keys=[x.shape[0] for x in train_data_loader.dataset]
            )
        valid_data_loader = convert_to_bucketed_dataloader(
            valid_data_loader, keys=[x.shape[0] for x in valid_data_loader.dataset], shuffle_same_key=False
        )
        return train_data_loader, valid_data_loader

    def init_model(self):
        self.model = ExplainableRecurrentPointProcess(n_types=self.n_types, **{**self.CONFIG.model})
        self.model = self.model.to(self.device)

    def train(self, train_data_loader, valid_data_loader):
        configs = self.CONFIG.train
        optimizer = getattr(torch.optim, configs.optimizer)(
            self.model.parameters(), lr=configs.lr
        )

        self.model.train()

        best_metric = float('inf')
        best_epoch = 0

        loss = {loss_type: MetricTracker(metric=loss_type) for loss_type in ['train', 'valid']}
        for epoch in range(configs.epochs):
            train_metrics, valid_metrics = self.model.train_epoch(
                train_data_loader,
                optimizer,
                valid_data_loader,
                device=self.device,
                **configs,
            )
            loss['train'].update(train_metrics['nll'].avg, n=train_metrics['nll'].count)
            loss['valid'].update(valid_metrics['nll'].avg, n=valid_metrics['nll'].count)

            msg = f'[Training] Epoch={epoch}'
            for k, v in train_metrics.items():
                msg += f', {k}={v.avg:.4f}'
            print(msg)  # logger.info(msg)
            msg = f'[Validation] Epoch={epoch}'
            for k, v in valid_metrics.items():
                msg += f', {k}={v.avg:.4f}'
            print(msg)  # logger.info(msg)

            if valid_metrics[configs.tune_metric].avg < best_metric:
                best_epoch = epoch
                best_metric = valid_metrics[configs.tune_metric].avg
                save_pytorch_model(self.model, self.bucket, self.tenant_id)

            if epoch - best_epoch >= configs.patience:
                print(f'Stopped training early at epoch {epoch}: ' +
                      f'Failed to improve validation {configs.tune_metric} in last {configs.patience} epochs')
                break

        # Reset model to the last-saved (best) version of the model
        self.model = load_pytorch_object(self.bucket, self.tenant_id, 'model')
        history = pd.DataFrame({
            'epoch': range(epoch + 1),
            'train': loss['train'].values,
            'valid': loss['valid'].values,
        })
        return history

    @staticmethod
    def plot_training_loss(history, tune_metric):
        plt.plot(history['train'], label='train')
        plt.plot(history['valid'], label='valid')
        plt.xlabel('epoch')
        plt.ylabel(tune_metric)
        plt.yscale('log')
        plt.legend()
        plt.show()

    def plot_precision_at_k(self, metrics):
        ks = self.CONFIG.model.ks
        precisions = [metrics[f'precision_at_{k}'].avg for k in ks]
        plt.plot(ks, precisions)
        plt.xlabel('Intensity')
        plt.ylabel('Precision @ Intensity')
        plt.show()

    def calculate_test_metrics(self, event_seqs):
        data_loader = DataLoader(
            EventSeqDataset(event_seqs), shuffle=False, **self.data_loader_args
        )
        metrics = self.model.evaluate(data_loader, device=self.device)
        msg = '[Test] ' + ', '.join(f'{k}={v.avg:.4f}' for k, v in metrics.items())
        print(msg)  # logger.info(msg)
        return metrics

    def predict(self, event_seqs, days=range(1, 31)):
        data_loader = DataLoader(
            EventSeqDataset(event_seqs), shuffle=False, **self.data_loader_args
        )
        intensities, cumulants, log_basis_weights = self.model.predict_event_intensities(data_loader, self.device, days)

        save_pytorch_dataset(intensities, self.bucket, self.tenant_id, 'event_intensities')
        save_pytorch_dataset(cumulants, self.bucket, self.tenant_id, 'event_cumulants')
        save_pytorch_dataset(log_basis_weights, self.bucket, self.tenant_id, 'log_basis_weights')

        return intensities, cumulants, log_basis_weights

    def evaluate_event_prediction(self):
        # Function to evaluate predictions for a specific event type
        pass

from functools import partial
import numpy as np
import torch
from torch import nn
import torch.nn.functional as f
from torch import FloatTensor, BoolTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

from upsell.basis_functions import Normal, Unity
from upsell.explain import batch_grad
from upsell.metric_tracker import MetricTracker
from upsell.utils.env import set_eval_mode
from upsell.utils.data_loader import generate_sequence_mask


class Decoder(nn.Module):
    def __init__(self, in_features, out_features, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = in_features

        self.net1 = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features),
        )
        if hidden_size == out_features:
            self.net2 = lambda x: x
        else:
            self.net2 = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.net1(x) + self.net2(x)


class ExplainableRecurrentPointProcess(nn.Module):
    def __init__(
            self,
            n_event_types: int,
            # Neural networks
            embedding_dim: int = 32,
            hidden_size: int = 32,
            rnn: str = 'GRU',
            dropout: float = 0.0,
            max_log_basis_weight: float = 30.0,
            # Basis functions
            basis_type: str = 'normal',
            n_bases: Optional[int] = None,
            max_mean: Optional[float] = None,
            basis_means: Optional[List[Union[int, float]]] = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_event_types = n_event_types

        # Basis functions
        if basis_type in ['equal', 'dyadic'] and (n_bases is None or max_mean is None):
            raise ValueError(f'n_bases and max_mean must be specified when basis_type={basis_type}')

        if basis_type == 'means' and basis_means is None:
            raise ValueError(f'basis_means must be specified when basis_type={basis_type}')

        self.n_bases = n_bases if n_bases is not None else len(basis_means)
        self.bases = self.define_basis_functions(basis_type, max_mean=max_mean, basis_means=basis_means)

        # Neural networks
        # TODO: move clamping into model
        self.max_log_basis_weight = max_log_basis_weight
        self.embedder, self.encoder, self.dropout, self.decoder = None, None, None, None
        self.define_model(embedding_dim, hidden_size, rnn, dropout)

    def define_model(self, embedding_dim: int, hidden_size: int, rnn: str, dropout: float) -> None:
        # TODO: Move this into nn.Sequential
        self.embedder = nn.Linear(self.n_event_types, embedding_dim, bias=False)
        self.encoder = getattr(nn, rnn)(
            input_size=embedding_dim + 1,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.decoder = Decoder(hidden_size, self.n_event_types * (self.n_bases + 1))

    def define_basis_functions(self, basis_type: str, max_mean=None, basis_means=None) -> nn.ModuleList:
        bases = [Unity()]
        if basis_type == 'equal':
            x = max_mean / (self.n_bases - 1)
            mu = [i * x for i in range(self.n_bases)]
            sigma = [x] * self.n_bases
        elif basis_type == 'dyadic':
            mu = [0] + sorted([max_mean / 2 ** i for i in range(self.n_bases-1)])
            sigma = [max_mean / 2 ** (self.n_bases-1) / 3] + [mu / 3 for mu in mu[1:]]
        elif basis_type == 'means':
            mu = sorted(basis_means)
            if mu[0] == 0:
                sigma = [mu[1] / (2 * 3)] + [mu / 3 for mu in mu[1:]]
            else:
                sigma = [mu / 3 for mu in mu]
        else:
            raise ValueError(f'Unrecognized basis_type={basis_type}')

        bases.append(Normal(mu=mu, sigma=sigma))
        return nn.ModuleList(bases)

    def forward(
        self,
        event_seqs: FloatTensor,
        return_weights: bool = True,
        target_type: Optional[Union[int, List[int]]] = None
    ) -> Union[FloatTensor, Tuple[FloatTensor, FloatTensor]]:
        """

        :param event_seqs: (Tensor) shape=[batch_size, time_stamps, 1 + n_event_types]
        :param return_weights: (bool) whether to return the basis weights
        :param target_type: (int, optional) specific event type index to predict
        :return:
            log_intensities (Tensor): shape=[batch_size, time_stamps, n_event_types],
                log conditional intensities evaluated at each timestamp for each event type
                (i.e. starting at t1).
            weights (Tensor, optional): shape=[batch_size, time_stamps, n_event_types, n_bases],
                basis weights evaluated at each previous event (i.e., starting at t0).
                Returned only when `return_weights` is `True`.
        """
        assert event_seqs.size(-1) == 1 + self.n_event_types, event_seqs.size()
        if target_type is not None:
            assert 0 <= target_type < self.n_event_types, target_type

        batch_size, time_stamps = event_seqs.size()[:2]

        # GET TIME FEATURES
        # Timestamps [batch_size, time_stamps + 1]:
        # (t0=0, t1, t2, ..., t_n)
        ts = f.pad(event_seqs[:, :, 0], (1, 0))
        # Time deltas [batch_size, time_stamps + 1]:
        # (0, t1 - t0, ..., t_{n} - t_{n - 1})
        dt = f.pad(ts[:, 1:] - ts[:, :-1], (1, 0))
        # Excl final time delta [batch_size, time_stamps, 1]:
        # (0, t1 - t0, ..., t_{n - 1} - t_{n - 2})
        time_feat = dt[:, :-1].unsqueeze(-1)

        # EMBED EVENT VECTOR AT EACH TIMESTAMP EXCEPT LAST
        # In: [batch_size, time_stamps - 1, n_event_types]
        # Out: [batch_size, time_stamps - 1, embedding_dim]
        # (0, z_1, ..., z_{n - 1})
        event_emb = self.embedder(event_seqs[:, :-1, 1:])

        # Add time deltas back in: [batch_size, time_stamps, embedding_dim + 1]
        type_feat = f.pad(event_emb, (0, 0, 1, 0))
        feat = torch.cat([time_feat, type_feat], dim=-1)

        # ENCODE EVENT HISTORY PER ACCOUNT
        # In: [batch_size, time_stamps, embedding_dim + 1]
        # Out: [batch_size, hidden_size]
        event_hist_emb, *_ = self.encoder(feat)

        # [batch_size, hidden_size]
        event_hist_emb_w_dropout = self.dropout(event_hist_emb)

        # DECODE INTO WEIGHTS FOR EACH BASIS FUNCTION
        # In: [batch_size, hidden_size]
        # Out: [batch_size, time_stamps, n_event_types, n_bases]
        raw_log_basis_weights = self.decoder(event_hist_emb_w_dropout).view(
            batch_size, time_stamps, self.n_event_types, -1
        )

        # Cap basis weights to avoid numerical instability
        # TODO: Can the clamping be defined in self.define_model()?
        #  - So that it's not redefined on every forward pass
        #  - So we don't have to make model configs like self.max_log_basis_weight a class attribute
        #  - Requires using a clamping function from torch.nn.functional or subclassing nn.Module
        log_basis_weights = torch.clamp(raw_log_basis_weights, max=self.max_log_basis_weight)

        if target_type is not None:
            log_basis_weights = log_basis_weights[:, :, target_type: target_type + 1]

        # GET CONDITIONAL INTENSITIES FOR EACH EVENT TYPE AT EACH TIMESTAMP
        # [batch_size, time_stamps, 1, n_bases]
        log_basis_probas = torch.cat(
            [basis.log_prob(dt[:, 1:, None]) for basis in self.bases], dim=2
        ).unsqueeze(-2)

        # [batch_size, time_stamps, n_event_types]
        log_intensities = (log_basis_weights + log_basis_probas).logsumexp(dim=-1)

        if return_weights:
            return log_intensities, log_basis_weights
        else:
            return log_intensities

    def _eval_cumulants(self, batch: FloatTensor, log_basis_weights: FloatTensor) -> FloatTensor:
        """
        Evaluate the cumulants (i.e., integral of CIFs at each location)
        """
        # (t1, t2, ..., t_n)
        ts = batch[:, :, 0]
        # (t1 - t0, ..., t_n - t_{n - 1})
        dt = (ts - f.pad(ts[:, :-1], (1, 0))).unsqueeze(-1)

        # [batch_size, time_stamps, n_bases]
        integrals = torch.cat([
            basis.cdf(dt) - basis.cdf(torch.zeros_like(dt)) for basis in self.bases
        ], dim=-1)

        # [batch_size, time_stamps, n_event_types]
        cumulants = integrals.unsqueeze(2).mul(log_basis_weights.exp()).sum(-1)
        return cumulants

    def _eval_nll(self, batch: FloatTensor, log_intensities: FloatTensor,
                  log_basis_weights: FloatTensor, mask: BoolTensor, debug: bool = False) -> float:
        """
        Evaluate the negative log loss at each timestamp
        1. Intensity of the events that actually occurred at t_i+1
        2. Intensity of no other events occurring between t_i and t_i+1
        """
        # Intensity of the events that actually occurred at t_i+1
        # (minimize the negative so that we maximize the intensity of these events)
        loss_part1 = sum([
            # Multiply the intensity for the account+time+event_type by the weight of the event type
            #   --> weight could represent the number of times the event occurred, number of keywords, etc.
            #   Note: log(intensity * weight) = log_intensity + log_weight
            -(log_intensities[i][j][k] + batch[i][j][k+1].log())
            for i, j, k in
            torch.nonzero(batch[:, :, 1:])  # list of indices of events that occurred
        ])

        # Intensity of no other events occurring between t_i and t_i+1
        # (minimize so the model doesn't predict events that didn't occur)
        loss_part2 = (
            self._eval_cumulants(batch, log_basis_weights)
            .sum(-1)
            .masked_select(mask)
            .sum()
        )
        if debug:
            print('loss_part1', loss_part1)
            print('loss_part2', loss_part2)
            print('batch.size(0)', batch.size(0))
            print('nll', (loss_part1 + loss_part2) / batch.size(0))

        return (loss_part1 + loss_part2) / batch.size(0)

    @staticmethod
    def _eval_avg_incidence_at_k(batch: FloatTensor, log_intensities: FloatTensor, k: float = 1.0, eps: float = 0.1)\
            -> Tuple[float, int]:
        # Select events with (k - eps) <= intensity <= (k + eps)
        mask = (log_intensities >= np.log(k - eps)) & (log_intensities < np.log(k + eps))
        events_w_intensity_k = batch[:, :, 1:].masked_select(mask)

        # Check average incidence of events with (k - eps) <= intensity <= (k + eps)
        avg_incidence = events_w_intensity_k.mean()
        n = len(events_w_intensity_k)
        return avg_incidence, n

    def train_epoch(
            self,
            train_data_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            valid_data_loader: Optional[DataLoader] = None,
            device: Optional = None,
            **kwargs,
    ) -> Tuple[Dict[str, MetricTracker], Dict[str, MetricTracker]]:
        train_metrics = {m: MetricTracker(metric=m) for m in ['nll', 'loss', 'l2_reg']}

        self.train()
        for batch in train_data_loader:
            if device:
                batch = batch.to(device)
            seq_length = (batch.abs().sum(-1) > 0).sum(-1)
            mask = generate_sequence_mask(seq_length)

            log_intensities, log_basis_weights = self.forward(
                batch, return_weights=True
            )
            nll = self._eval_nll(
                batch, log_intensities, log_basis_weights, mask
            )
            if kwargs['l2_reg'] > 0:
                l2_reg = (
                        kwargs['l2_reg']
                        * log_basis_weights.permute(2, 3, 0, 1)
                        .masked_select(mask)
                        .exp()
                        .pow(2)
                        .mean()
                )
            else:
                l2_reg = 0.0
            loss = nll + l2_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metrics['loss'].update(loss, batch.size(0))
            train_metrics['nll'].update(nll, batch.size(0))
            train_metrics['l2_reg'].update(l2_reg, seq_length.sum())

        if valid_data_loader:
            valid_metrics = self.evaluate(valid_data_loader, device=device)
        else:
            valid_metrics = None

        return train_metrics, valid_metrics

    def evaluate(self, data_loader: DataLoader, event_index: Optional[int] = None,
                 device: Optional[torch.device] = None) -> Dict[str, MetricTracker]:
        metrics = {m: MetricTracker(metric=m) for m in ['nll']}

        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                if device:
                    batch = batch.to(device)

                seq_length = (batch.abs().sum(-1) > 0).sum(-1)
                mask = generate_sequence_mask(seq_length)

                log_intensities, log_basis_weights = self.forward(
                    batch, return_weights=True, target_type=event_index
                )
                # TODO: calculate NLL for a specific event type
                nll = self._eval_nll(
                    batch, log_intensities, log_basis_weights, mask
                )

                metrics['nll'].update(nll, batch.size(0))

        return metrics

    def generate_calibration_curve(self, data_loader: DataLoader, ks: list[float], event_index: Optional[int] = None,
                                   eps: float = 0.1, device: Optional[torch.device] = None) -> Dict[str, MetricTracker]:
        avg_incidences = {k: MetricTracker(metric=k) for k in ks}

        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                if device:
                    batch = batch.to(device)

                log_intensities, log_basis_weights = self.forward(
                    batch, return_weights=True, target_type=event_index
                )

                for k in ks:
                    p, n = self._eval_avg_incidence_at_k(batch, log_intensities, k, eps)
                    if n > 0:
                        avg_incidences[k].update(p, n)

        return avg_incidences

    def predict_future_event_intensities(
            self,
            data_loader: DataLoader,
            device: Optional = None,
            time_steps: List[int] = range(1, 31)
    ) -> Tuple[FloatTensor, FloatTensor]:
        batch_intensities, batch_cumulants = [], []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                if device:
                    batch = batch.to(device)

                seq_length = (batch.abs().sum(-1) > 0).sum(-1)
                _, log_basis_weights = self.forward(batch, return_weights=True)

                # Get the basis weights for the last event vector in each account [batch_size, n_event_types, n_bases]
                last_log_basis_weights = torch.cat(
                    [log_basis_weights[i][j-1].unsqueeze(0) for i, j in enumerate(seq_length)])

                dt = torch.Tensor([[day] for day in time_steps])

                # PDF -> INTENSITIES
                # Get the probability for time = dt from each basis function [time_steps, n_bases]
                log_probas = torch.cat([basis.log_prob(dt).type(torch.float32) for basis in self.bases], dim=1)

                # Multiply probas by basis weights [batch_size, n_event_types, time_steps]
                intensities = np.exp(last_log_basis_weights.unsqueeze(2) + log_probas).sum(-1)

                # CDF -> CUMULANTS
                # Integrate each basis function from 0 to t [time_steps, n_bases]
                integrals = torch.cat([basis.cdf(dt).type(torch.float32) - basis.cdf(torch.zeros_like(dt))
                                       for basis in self.bases], dim=-1)

                # Multiply integrals by basis weights [batch_size, n_event_types, time_steps]
                cumulants = integrals.mul(last_log_basis_weights.unsqueeze(2).exp()).sum(-1)

                batch_intensities.append(intensities)
                batch_cumulants.append(cumulants)

        # Stack intensities and cumulants [all accounts, n_event_types, days]
        all_intensities = torch.cat(batch_intensities, dim=0)
        all_cumulants = torch.cat(batch_cumulants, dim=0)

        return all_intensities, all_cumulants

    def get_infectivity(
            self,
            data_loader: DataLoader,
            device: Optional = None,
            steps: int = 50,
            occurred_type_only: bool = False,
    ) -> FloatTensor:
        def get_attribution_target(x, target_type):
            # Attribution target: cumulative intensity from t_i to t_i+1
            _, log_basis_weights = self.forward(
                x, return_weights=True, target_type=target_type
            )
            cumulants = self._eval_cumulants(x, log_basis_weights)
            # Drop first timestamp as it corresponds to intensity between (t_0, t_1)
            return cumulants[:, 1:]

        # Set module.training = False for Dropout layer and True for all other layers
        set_eval_mode(self)

        # Freeze the model parameters to reduce unnecessary backpropagation
        for param in self.parameters():
            param.requires_grad_(False)

        # Track attribution for each event type on other event types
        # attr_matrix[i, j] = the event contribution of the j-th event type to the cumulative intensity
        #     prediction of the i-th event type, relative to the baseline of 0.
        # Shape: [n_event_types, n_event_types]
        attr_matrix = torch.zeros(self.n_event_types, self.n_event_types, device=device)

        # Track occurrences of each event type. Shape: [n_event_types]
        type_counts = torch.zeros(self.n_event_types, device=device)

        for batch in tqdm(data_loader):
            if device:
                batch = batch.to(device)

            batch_size, time_stamps = batch.size()[:2]
            seq_lengths = (batch.abs().sum(-1) > 0).sum(-1)

            # Baseline = timestamps with a vector of zeros
            # Size: [batch_size, time_stamps, n_event_types+1]
            baselines = f.pad(batch[:, :, :1], (0, self.n_event_types))

            # [batch_size, time_stamps-1]
            mask = generate_sequence_mask(seq_lengths - 1, device=device).unsqueeze(-1)

            if occurred_type_only:
                # Get indexes of event types that occurred in the batch
                occurred_types = set(batch[:, :, 1:].nonzero()[:, -1].tolist())
            else:
                # Get indexes of all event types
                occurred_types = range(self.n_event_types)

            # Track gradient for each event type. Shape: [n_event_types, batch_size, time_stamps-1]
            event_scores = torch.zeros(self.n_event_types, batch_size, time_stamps - 1, device=device)

            # Multiply event weights w.r.t. baseline by a range of numbers between 0 and 1.
            # Then stack in ascending order:
            #     [[inputs * 0], [inputs * 1/(steps-1)], [inputs * 2/(steps-1)], ..., [inputs * 1]]
            # Shape: [steps * batch_size, time_steps, n_event_types+1]
            scaled_batch = (
                baselines.unsqueeze(dim=-1) +
                (batch - baselines).unsqueeze(dim=-1) * torch.linspace(0, 1, steps, device=batch.device)
            )\
                .permute([-1, *range(batch.ndimension())])\
                .contiguous()\
                .view(-1, *batch.size()[1:])

            for k in occurred_types:
                grads = batch_grad(
                    partial(get_attribution_target, target_type=k),
                    scaled_batch,
                    mask
                )

                # Unpack steps
                # Shape: [steps, batch_size, time_stamps, n_event_types+1]
                grads = grads.view(steps, batch_size, *grads.size()[1:])

                # Average gradients across all steps (except the first, where inputs are all zero)
                # Shape: [batch_size, time_stamps, n_event_types+1]
                avg_grads = grads[1:].mean(dim=0)

                # "Integrate" gradients over the difference between inputs and baseline
                # Shape: [batch_size, time_stamps, n_event_types+1]
                integrated_grads = (batch - baselines) * avg_grads

                # For each account and timestamp (excl final time step), sum gradients from each event type
                event_scores[k] = integrated_grads[:, :-1].sum(-1)

            # Update attribution matrix with the gradient of each event type w.r.t. each event type
            in_events = batch[:, :-1, 1:]
            for b, t, k in in_events.nonzero().tolist():
                # attr_matrix[:, k] = impact of event k on all event types
                # event_scores[:, b, t] = gradient of each event type at the account + timestamp where event k occurred
                # Multiply the gradient by the weight of the event k at account b + timestamp t
                attr_matrix[:, k] += (in_events[b, t, k] * event_scores[:, b, t])

            # Sum the weights for each event type across timestamps and accounts
            type_counts += batch[:, :, 1:].sum([0, 1])

        # Plus one to avoid division by zero
        attr_matrix /= type_counts[None, :] + 1

        return attr_matrix.detach().cpu()

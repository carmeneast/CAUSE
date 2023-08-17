from functools import partial
import numpy as np
import torch
from torch import nn
import torch.nn.functional as f
from tqdm import tqdm

from upsell.utils import batch_integrated_gradient, generate_sequence_mask, set_eval_mode
from upsell.basis_functions import Normal, Unity
from upsell.metric_tracker import MetricTracker


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
            rnn: str = "GRU",
            dropout: float = 0.0,
            max_log_basis_weight: float = 30.0,
            # Basis functions
            basis_type: str = "normal",
            n_bases: int = None,
            max_mean: float = None,
            basis_means: list = None,
            # Metrics
            ks: list = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_event_types = n_event_types

        # Basis functions
        if (n_bases is None or max_mean is None) and basis_means is None:
            raise ValueError('Either (n_bases and max_mean) or basis_means must be specified.')

        if n_bases is None:
            n_bases = len(basis_means)
        self.bases = self.define_basis_functions(basis_type, n_bases, max_mean=max_mean,
                                                 basis_means=basis_means)

        # Neural networks
        # TODO: move clamping into model
        self.max_log_basis_weight = max_log_basis_weight
        self.embedder, self.encoder, self.dropout, self.decoder = self.define_model(
            embedding_dim, hidden_size, rnn, dropout, n_bases
        )

        # Metrics
        if ks is None:
            self.ks = [0.1, 1.0]
        else:
            self.ks = ks
        self.metrics = ['nll'] + [f'precision_at_{k}' for k in self.ks]

    def define_model(self, embedding_dim, hidden_size, rnn, dropout, n_bases):
        # TODO: Move this into nn.Sequential
        embedder = nn.Linear(self.n_event_types, embedding_dim, bias=False)
        encoder = getattr(nn, rnn)(
            input_size=embedding_dim + 1,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
        )
        dropout = nn.Dropout(p=dropout)
        decoder = Decoder(hidden_size, self.n_event_types * (n_bases + 1))
        return embedder, encoder, dropout, decoder

    @staticmethod
    def define_basis_functions(basis_type, n_bases, max_mean=None, basis_means=None):
        bases = [Unity()]
        if basis_type == 'equal':
            mu, sigma = [], []
            for i in range(n_bases):
                mu.append(i * max_mean / (n_bases - 1))
                sigma.append(max_mean / (n_bases - 1))
        elif basis_type == 'dyadic':
            if basis_means is not None:
                mu = sorted(basis_means)
                sigma = [mu[1] / (2 * 3)] + [mu / 3 for mu in mu[1:]]
            else:
                mu = [0] + sorted([max_mean / 2 ** i for i in range(n_bases-1)])
                sigma = [max_mean / 2 ** (n_bases-1) / 3] + [mu / 3 for mu in mu[1:]]
        else:
            raise ValueError(f'Unrecognized basis_type={basis_type}')

        bases.append(Normal(mu=mu, sigma=sigma))
        return nn.ModuleList(bases)

    def forward(self, event_seqs, return_weights=True, target_type=None):
        """[summary]

        Args:
          event_seqs (Tensor): shape=[batch_size, time_stamps, 1 + n_event_types]

          return_weights (bool): whether to return the basis weights.

          target_type (int): whether to only predict for a specific type

        Returns:
           log_intensities (Tensor): shape=[batch_size, time_stamps, n_event_types],
             log conditional intensities evaluated at each event for each type
             (i.e. starting at t1).
           weights (Tensor, optional): shape=[batch_size, time_stamps, n_event_types, n_bases],
             basis weights evaluated at each previous event (i.e., starting at t0).
             Returned only when `return_weights` is `True`.

        """
        assert event_seqs.size(-1) == 1 + self.n_event_types, event_seqs.size()
        if target_type:
            assert 0 <= target_type < self.n_event_types, target_type

        batch_size, time_stamps = event_seqs.size()[:2]

        # (t0=0, t1, t2, ..., t_n)
        ts = f.pad(event_seqs[:, :, 0], (1, 0))
        # (0, t1 - t0, ..., t_{n} - t_{n - 1})
        dt = f.pad(ts[:, 1:] - ts[:, :-1], (1, 0))
        # (0, t1 - t0, ..., t_{n - 1} - t_{n - 2})
        temp_feat = dt[:, :-1].unsqueeze(-1)

        # (0, z_1, ..., z_{n - 1})
        type_feat = self.embedder(event_seqs[:, :-1, 1:])
        type_feat = f.pad(type_feat, (0, 0, 1, 0))

        feat = torch.cat([temp_feat, type_feat], dim=-1)
        history_emb, *_ = self.encoder(feat)
        history_emb = self.dropout(history_emb)

        # [batch_size, time_stamps, n_event_types, n_bases]
        log_basis_weights = self.decoder(history_emb).view(
            batch_size, time_stamps, self.n_event_types, -1
        )
        log_basis_weights = torch.clamp(log_basis_weights, max=self.max_log_basis_weight)

        if target_type:
            log_basis_weights = log_basis_weights[:, :, target_type: target_type + 1]

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

    def _eval_cumulants(self, batch, log_basis_weights):
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

    def _eval_nll(self, batch, log_intensities, log_basis_weights, mask, debug=False):
        """
        Evaluate the negative log loss at each location
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
    def _eval_precision_at_k(batch, log_intensities, k=1.0):
        # Select events with intensity >= k
        mask = log_intensities >= np.log(k)
        high_intensity_events = batch[:, :, 1:].masked_select(mask)

        # Check which events actually occurred (ignoring event weights)
        precision = (high_intensity_events > 0).float().mean()
        n = len(high_intensity_events)
        return precision, n

    def train_epoch(
            self,
            train_data_loader,
            optim,
            valid_data_loader=None,
            device=None,
            **kwargs,
    ):
        self.train()

        train_metrics = {m: MetricTracker(metric=m) for m in ['loss', 'l2_reg'] + self.metrics}

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
            if kwargs["l2_reg"] > 0:
                l2_reg = (
                        kwargs["l2_reg"]
                        * log_basis_weights.permute(2, 3, 0, 1)
                        .masked_select(mask)
                        .exp()
                        .pow(2)
                        .mean()
                )
            else:
                l2_reg = 0.0
            loss = nll + l2_reg

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_metrics['loss'].update(loss, batch.size(0))
            train_metrics['nll'].update(nll, batch.size(0))
            train_metrics['l2_reg'].update(l2_reg, seq_length.sum())
            for k in self.ks:
                p, n = self._eval_precision_at_k(batch, log_intensities, k)
                train_metrics[f'precision_at_{k}'].update(p, n)

        if valid_data_loader:
            valid_metrics = self.evaluate(valid_data_loader, device=device)
        else:
            valid_metrics = None

        return train_metrics, valid_metrics

    def evaluate(self, data_loader, device=None):
        metrics = {m: MetricTracker(metric=m) for m in self.metrics}

        self.eval()
        with torch.no_grad():
            for batch in data_loader:
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

                metrics['nll'].update(nll, batch.size(0))
                for k in self.ks:
                    p, n = self._eval_precision_at_k(batch, log_intensities, k)
                    metrics[f'precision_at_{k}'].update(p, n)

        return metrics

    def predict_future_event_intensities(self, data_loader, device, time_steps=range(1, 31)):
        batch_intensities, batch_cumulants, batch_log_basis_weights = [], [], []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                if device:
                    batch = batch.to(device)

                seq_length = (batch.abs().sum(-1) > 0).sum(-1)
                _, log_basis_weights = self.forward(batch, return_weights=True)

                # Get the basis weights for the last event vector in each account [batch_size, n_event_types, n_bases]
                last_log_basis_weights = torch.cat(
                    [log_basis_weights[i][j-1].unsqueeze(0) for i, j in enumerate(seq_length)])

                dt = torch.tensor([[day] for day in time_steps])

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
                batch_log_basis_weights.append(last_log_basis_weights)

        # Stack intensities and cumulants [all accounts, n_event_types, days]
        all_intensities = torch.cat(batch_intensities, dim=0)
        all_cumulants = torch.cat(batch_cumulants, dim=0)

        # Stack basis weights [all accounts, n_event_types, n_bases]
        all_log_basis_weights = torch.cat(batch_log_basis_weights, dim=0)

        return all_intensities, all_cumulants, all_log_basis_weights

    def get_infectivity(
            self,
            data_loader,
            device=None,
            steps=50,
            occurred_type_only=False,
    ):
        def func(x, target_type):
            _, log_basis_weights = self.forward(
                x, return_weights=True, target_type=target_type
            )
            cumulants = self._eval_cumulants(x, log_basis_weights)
            # drop index=0 as it corresponds to (t_0, t_1)
            return cumulants[:, 1:]

        set_eval_mode(self)
        # freeze the model parameters to reduce unnecessary backpropagation.
        for param in self.parameters():
            param.requires_grad_(False)

        A = torch.zeros(self.n_event_types, self.n_event_types, device=device)
        type_counts = torch.zeros(self.n_event_types, device=device).long()

        for batch in tqdm(data_loader):
            if device:
                batch = batch.to(device)

            batch_size, time_stamps = batch.size()[:2]
            seq_lengths = (batch.abs().sum(-1) > 0).sum(-1)

            inputs = torch.cat(
                [
                    batch[:, :, :1],
                    f.one_hot(batch[:, :, 1].long(), self.n_event_types).float(),
                ],
                dim=-1,
            )
            baselines = f.pad(inputs[:, :, :1], (0, self.n_event_types))
            mask = generate_sequence_mask(seq_lengths - 1, device=device)

            if occurred_type_only:
                occurred_types = set(
                    batch[:, :, 1]
                    .masked_select(generate_sequence_mask(seq_lengths))
                    .long()
                    .tolist()
                )
            else:
                occurred_types = range(self.n_event_types)

            event_scores = torch.zeros(
                self.n_event_types, batch_size, time_stamps - 1, device=device
            )
            for k in occurred_types:
                ig = batch_integrated_gradient(
                    partial(func, target_type=k),
                    inputs,
                    baselines=baselines,
                    mask=mask.unsqueeze(-1),
                    steps=steps,
                )
                event_scores[k] = ig[:, :-1].sum(-1)

            # shape=[n_event_types, batch_size, time_stamps-1]
            A.scatter_add_(
                1,
                index=batch[:, :-1, 1]
                .long()
                .view(1, -1)
                .expand(self.n_event_types, -1),
                src=event_scores.view(self.n_event_types, -1),
            )

            ks = (
                batch[:, :, 1]
                .long()
                .masked_select(generate_sequence_mask(seq_lengths))
            )
            type_counts.scatter_add_(0, index=ks, src=torch.ones_like(ks))

        # plus one to avoid division by zero
        A /= type_counts[None, :].float() + 1

        return A.detach().cpu()

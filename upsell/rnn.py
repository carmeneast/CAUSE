from collections import defaultdict
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm

from pkg.explain.integrated_gradient import batch_integrated_gradient
from pkg.utils.misc import AverageMeter
from pkg.utils.torch import ResidualLayer, generate_sequence_mask, set_eval_mode
from pkg.models.func_basis import Normal, Unity


class ExplainableRecurrentPointProcess(nn.Module):
    def __init__(
            self,
            n_types: int,
            # Neural networks
            embedding_dim: int = 32,
            hidden_size: int = 32,
            rnn: str = "GRU",
            dropout: float = 0.0,
            max_log_basis_weight: float = 30.0,
            nll_apply_event_weighting: bool = True,
            # Basis functions
            basis_type: str = "normal",
            n_bases: int = None,
            max_mean: float = None,
            basis_means: list = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_types = n_types
        self.nll_apply_event_weighting = nll_apply_event_weighting
        self.max_log_basis_weight = max_log_basis_weight

        if (n_bases is None or max_mean is None) and basis_means is None:
            raise ValueError('Either (n_bases and max_mean) or basis_means must be specified.')

        if n_bases is None:
            n_bases = len(basis_means)
        self.bases = self.define_basis_functions(basis_type, n_bases, max_mean=max_mean,
                                                 basis_means=basis_means)

        self.embedder = nn.Linear(n_types, embedding_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.encoder = getattr(nn, rnn)(
            input_size=embedding_dim + 1,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder = ResidualLayer(hidden_size, n_types * (n_bases + 1))

    @staticmethod
    def define_basis_functions(basis_type, n_bases, max_mean=None, basis_means=None):
        bases = [Unity()]
        if basis_type == 'equal':
            loc, scale = [], []
            for i in range(n_bases):
                loc.append(i * max_mean / (n_bases - 1))
                scale.append(max_mean / (n_bases - 1))
        elif basis_type == 'dyadic':
            if basis_means is not None:
                loc = sorted(basis_means)
                scale = [loc[1] / (2 * 3)] + [mu / 3 for mu in loc[1:]]
            else:
                loc = [0] + sorted([max_mean / 2 ** i for i in range(n_bases-1)])
                scale = [max_mean / 2 ** (n_bases-1) / 3] + [mu / 3 for mu in loc[1:]]
        else:
            raise ValueError(f'Unrecognized basis_type={basis_type}')

        bases.append(Normal(loc=loc, scale=scale))
        return nn.ModuleList(bases)

    def forward(self, event_seqs, onehot=True, need_weights=True, target_type=-1):
        """[summary]

        Args:
          event_seqs (Tensor): shape=[batch_size, T, 2]
            or [batch_size, T, 1 + n_types]. The last dimension
            denotes the timestamp and the type of event, respectively.

          onehot (bool): whether the event types are represented by one-hot
            vectors.

          need_weights (bool): whether to return the basis weights.

          target_type (int): whether to only predict for a specific type

        Returns:
           log_intensities (Tensor): shape=[batch_size, T, n_types],
             log conditional intensities evaluated at each event for each type
             (i.e. starting at t1).
           weights (Tensor, optional): shape=[batch_size, T, n_types, n_bases],
             basis weights intensities evaluated at each previous event (i.e.,
             tarting at t0). Returned only when `need_weights` is `True`.

        """
        assert event_seqs.size(-1) == 1 + (
            self.n_types if onehot else 1
        ), event_seqs.size()

        batch_size, T = event_seqs.size()[:2]

        # (t0=0, t1, t2, ..., t_n)
        ts = f.pad(event_seqs[:, :, 0], (1, 0))
        # (0, t1 - t0, ..., t_{n} - t_{n - 1})
        dt = f.pad(ts[:, 1:] - ts[:, :-1], (1, 0))
        # (0, t1 - t0, ..., t_{n - 1} - t_{n - 2})
        temp_feat = dt[:, :-1].unsqueeze(-1)

        # (0, z_1, ..., z_{n - 1})
        if onehot:
            type_feat = self.embedder(event_seqs[:, :-1, 1:])
        else:
            type_feat = self.embedder(
                f.one_hot(event_seqs[:, :-1, 1].long(), self.n_types).float()
            )
        type_feat = f.pad(type_feat, (0, 0, 1, 0))

        feat = torch.cat([temp_feat, type_feat], dim=-1)
        history_emb, *_ = self.encoder(feat)
        history_emb = self.dropout(history_emb)

        # [B, T, K or 1, R]
        log_basis_weights = self.decoder(history_emb).view(
            batch_size, T, self.n_types, -1
        )
        log_basis_weights = torch.clamp(log_basis_weights, max=self.max_log_basis_weight)
        if target_type >= 0:
            log_basis_weights = log_basis_weights[
                                :, :, target_type: target_type + 1
                                ]

        # [B, T, 1, R]
        basis_values = torch.cat(
            [basis.log_prob(dt[:, 1:, None]) for basis in self.bases], dim=2
        ).unsqueeze(-2)

        log_intensities = (log_basis_weights + basis_values).logsumexp(dim=-1)

        if need_weights:
            return log_intensities, log_basis_weights
        else:
            return log_intensities

    def _eval_cumulants(self, batch, log_basis_weights):
        """
        Evaluate the cumulants (i.e., integral of CIFs at each location)
        """
        ts = batch[:, :, 0]
        # (t1 - t0, ..., t_n - t_{n - 1})
        dt = (ts - f.pad(ts[:, :-1], (1, 0))).unsqueeze(-1)
        # [B, T, R]
        integrals = torch.cat(
            [
                basis.cdf(dt) - basis.cdf(torch.zeros_like(dt))
                for basis in self.bases
            ],
            dim=-1,
        )
        cumulants = integrals.unsqueeze(2).mul(log_basis_weights.exp()).sum(-1)
        return cumulants

    def _eval_nll(self, batch, log_intensities, log_basis_weights, mask, debug=False):
        # Probability of the events that actually occurred at t_i+1
        if self.nll_apply_event_weighting:
            loss_part1 = sum([
                # Multiply the intensity for the account+time+event_type by the weight of the event type
                #   --> weight could represent the number of times the event occurred, number of keywords, etc.
                -log_intensities[i][j][k] * batch[i][j][k+1]
                for i, j, k in
                torch.nonzero(batch[:, :, 1:])  # list of indices of events that occurred
            ])
        else:
            loss_part1 = sum([
                -log_intensities[i][j][k] for i, j, k in
                torch.nonzero(batch[:, :, 1:])  # list of indices of events that occurred
            ])

        # Probability that no other events occurred between t_i and t_i+1
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
    def _eval_acc(batch, intensities, mask):
        types_pred = intensities.argmax(dim=-1).masked_select(mask)
        types_true = batch[:, :, 1].long().masked_select(mask)
        return (types_pred == types_true).float().mean()

    def train_epoch(
            self,
            train_dataloader,
            optim,
            valid_dataloader=None,
            device=None,
            **kwargs,
    ):
        self.train()

        train_metrics = defaultdict(AverageMeter)

        for batch in train_dataloader:
            if device:
                batch = batch.to(device)
            seq_length = (batch.abs().sum(-1) > 0).sum(-1)
            mask = generate_sequence_mask(seq_length)

            log_intensities, log_basis_weights = self.forward(
                batch, need_weights=True
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

            train_metrics["loss"].update(loss, batch.size(0))
            train_metrics["nll"].update(nll, batch.size(0))
            train_metrics["l2_reg"].update(l2_reg, seq_length.sum())
            train_metrics["acc"].update(
                self._eval_acc(batch, log_intensities, mask), seq_length.sum()
            )

        if valid_dataloader:
            valid_metrics = self.evaluate(valid_dataloader, device=device)
        else:
            valid_metrics = None

        return train_metrics, valid_metrics

    def evaluate(self, dataloader, device=None):
        metrics = defaultdict(AverageMeter)

        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                if device:
                    batch = batch.to(device)

                seq_length = (batch.abs().sum(-1) > 0).sum(-1)
                mask = generate_sequence_mask(seq_length)

                log_intensities, log_basis_weights = self.forward(
                    batch, need_weights=True
                )
                nll = self._eval_nll(
                    batch, log_intensities, log_basis_weights, mask
                )

                metrics["nll"].update(nll, batch.size(0))
                metrics["acc"].update(
                    self._eval_acc(batch, log_intensities, mask),
                    seq_length.sum(),
                )

        return metrics

    def predict_event_intensities(self, data_loader, device, days=range(1, 31)):
        batch_intensities, batch_cumulants, batch_log_basis_weights = [], [], []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                if device:
                    batch = batch.to(device)

                seq_length = (batch.abs().sum(-1) > 0).sum(-1)
                _, log_basis_weights = self.forward(batch, need_weights=True)

                # Get the basis weights for the last event vector in each account [B, K, R]
                last_log_basis_weights = torch.cat(
                    [log_basis_weights[i][j-1].unsqueeze(0) for i, j in enumerate(seq_length)])

                dt = torch.tensor([[day] for day in days])

                # PDF -> INTENSITIES
                # Get the probability for day = dt from each basis function [days, R]
                log_probas = torch.cat([basis.log_prob(dt) for basis in self.bases], dim=1)

                # Multiply probas by basis weights [B, K, days]
                intensities = np.exp(last_log_basis_weights.unsqueeze(2) + log_probas).sum(-1)

                # CDF -> CUMULANTS
                # Integrate each basis function from 0 to t [days, R]
                integrals = torch.cat([basis.cdf(dt) - basis.cdf(torch.zeros_like(dt))
                                       for basis in self.bases], dim=-1)

                # Multiply integrals by basis weights [B, K, days]
                cumulants = integrals.mul(last_log_basis_weights.unsqueeze(2).exp()).sum(-1)

                batch_intensities.append(intensities)
                batch_cumulants.append(cumulants)
                batch_log_basis_weights.append(last_log_basis_weights)

        # Stack intensities and cumulants [all accounts, K, days]
        all_intensities = torch.cat(batch_intensities, dim=0)
        all_cumulants = torch.cat(batch_cumulants, dim=0)

        # Stack basis weights [all accounts, K, R]
        all_log_basis_weights = torch.cat(batch_log_basis_weights, dim=0)

        return all_intensities, all_cumulants, all_log_basis_weights

    def get_infectivity(
            self,
            dataloader,
            device=None,
            steps=50,
            occurred_type_only=False,
    ):
        def func(X, target_type):
            _, log_basis_weights = self.forward(
                X, onehot=True, need_weights=True, target_type=target_type
            )
            cumulants = self._eval_cumulants(X, log_basis_weights)
            # drop index=0 as it corresponds to (t_0, t_1)
            return cumulants[:, 1:]

        set_eval_mode(self)
        # freeze the model parameters to reduce unnecessary backpropagation.
        for param in self.parameters():
            param.requires_grad_(False)

        A = torch.zeros(self.n_types, self.n_types, device=device)
        type_counts = torch.zeros(self.n_types, device=device).long()

        for batch in tqdm(dataloader):
            if device:
                batch = batch.to(device)

            batch_size, T = batch.size()[:2]
            seq_lengths = (batch.abs().sum(-1) > 0).sum(-1)

            inputs = torch.cat(
                [
                    batch[:, :, :1],
                    f.one_hot(batch[:, :, 1].long(), self.n_types).float(),
                ],
                dim=-1,
            )
            baselines = f.pad(inputs[:, :, :1], (0, self.n_types))
            mask = generate_sequence_mask(seq_lengths - 1, device=device)

            if occurred_type_only:
                occurred_types = set(
                    batch[:, :, 1]
                    .masked_select(generate_sequence_mask(seq_lengths))
                    .long()
                    .tolist()
                )
            else:
                occurred_types = range(self.n_types)

            event_scores = torch.zeros(
                self.n_types, batch_size, T - 1, device=device
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

            # shape=[K, B, T - 1]
            A.scatter_add_(
                1,
                index=batch[:, :-1, 1]
                .long()
                .view(1, -1)
                .expand(self.n_types, -1),
                src=event_scores.view(self.n_types, -1),
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

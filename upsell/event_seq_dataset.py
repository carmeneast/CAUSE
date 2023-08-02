import torch
import torch.nn as nn
from torch.utils.data import Dataset


class EventSeqDataset(Dataset):
    """
    Dataset for storing event sequences.
    :param event_seqs: List of sparse matrices. Each row of the sparse matrix has a timestamp and one-hot encoded
     vector describing which event types occurred at the timestamp.
     Shape: accounts x timestamps x (event types + 1)
    :param min_length: int -> Minimum number of timestamps required for an event sequence to be included in the dataset
    :param sort_by_length: bool -> whether to sort the event sequences by length in descending order
    """

    def __init__(self, event_seqs, min_length=1, sort_by_length=False):
        self.min_length = min_length
        self._event_seqs = [
            torch.sparse_coo_tensor(
                indices=torch.Tensor([seq.row, seq.col]),
                values=seq.data,
                size=seq.shape,
                dtype=torch.float32
            ).to_dense()
            for seq in event_seqs
            if seq.shape[0] >= min_length
        ]
        if sort_by_length:
            self._event_seqs = sorted(self._event_seqs, key=lambda x: -len(x))

    def __len__(self):
        return len(self._event_seqs)

    def __getitem__(self, i):
        # TODO: can instead compute the elapsed time between events
        return self._event_seqs[i]

    @staticmethod
    def collate_fn(x):
        return nn.utils.rnn.pad_sequence(x, batch_first=True)

import random
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch import BoolTensor, LongTensor


def split_data_loader(data_loader: DataLoader, weights: List[float]) -> List[DataLoader]:
    """
    Split a data loader into two data loaders, each with a portion of the dataset in the original
    data loader. Useful for splitting into train and validation data loaders.
    """
    dataset = data_loader.dataset

    n = len(dataset)
    lengths = []
    pct = 1.0
    for weight in weights:
        lengths.append(int(n * weight / pct))
        n -= lengths[-1]
        pct -= weight
    assert n == 0 and pct == 0.0

    datasets = torch.utils.data.random_split(dataset, lengths)

    copied_fields = ['batch_size', 'num_workers', 'collate_fn', 'drop_last']
    data_loaders = []
    for d in datasets:
        data_loaders.append(
            DataLoader(
                dataset=d, **{k: getattr(data_loader, k) for k in copied_fields}
            )
        )
    return data_loaders


class KeyBucketedBatchSampler(torch.utils.data.Sampler):
    """
    Pseudo bucketed batch sampler.

    Sample in a way that puts similarly-sized records into the same batch.
    :param keys: List[int]. list of keys by which the same or nearby keys are allocated
        in the same or nearby batches.
    :param batch_size: int. Batch size.
    :param drop_last: bool. Whether to drop the last incomplete batch. Defaults to False.
    :param shuffle_same_key: bool. Whether to shuffle the instances of the same keys. Defaults to False.
    """

    def __init__(self, keys: List[int], batch_size: int, drop_last: bool = False, shuffle_same_key: bool = False):
        super().__init__(self)
        self.keys = keys
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_same_key = shuffle_same_key

        # bucket sort; maintain random order inside each bucket
        buckets = {}
        for i, key in enumerate(self.keys):
            if key not in buckets:
                buckets[key] = [i]
            else:
                buckets[key].append(i)

        self.buckets = buckets

    def __iter__(self):
        indices = []
        for key in sorted(self.buckets.keys()):
            v = self.buckets[key]
            if self.shuffle_same_key:
                random.shuffle(v)
            indices += v

        index_batches = []
        for i in range(0, len(indices), self.batch_size):
            j = min(i + self.batch_size, len(indices))
            index_batches.append(indices[i:j])
        del indices

        if self.drop_last and len(index_batches[-1]) < self.batch_size:
            index_batches = index_batches[:-1]

        random.shuffle(index_batches)
        for indices in index_batches:
            yield indices

    def __len__(self):
        if self.drop_last:
            return len(self.keys) // self.batch_size
        else:
            return (len(self.keys) + self.batch_size - 1) // self.batch_size


def convert_to_bucketed_data_loader(data_loader: DataLoader, key_fn: Optional[Callable] = None,
                                    keys: Optional[List] = None, shuffle_same_key: bool = True):
    """
    Convert a data loader to bucketed data loader with a given keys.

    Args:
        data_loader: The input data loader.
        key_fn: function to extract keys used for constructing
          the bucketed data loader; should be of the same key as the
          dataset.
        keys: list of keys used for sorting the elements in the dataset.
        shuffle_same_key: Whether to shuffle the instances of the same keys. Defaults to False.

    Returns:
        DataLoader:
    """

    assert (data_loader.batch_size is not None), \
        "The `batch_size` must be present for the input dataloader"

    dataset = data_loader.dataset

    if key_fn is not None and keys is None:
        keys = [key_fn(dataset[i]) for i in range(len(dataset))]
    elif keys is not None and key_fn is None:
        assert len(keys) == len(dataset)
    else:
        raise ValueError("Either `key_fn` or `keys` must be set.")

    batch_sampler = KeyBucketedBatchSampler(
        keys,
        batch_size=data_loader.batch_size,
        drop_last=data_loader.drop_last,
        shuffle_same_key=shuffle_same_key,
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=data_loader.collate_fn,
        num_workers=data_loader.num_workers,
    )


def generate_sequence_mask(lengths: LongTensor, device: Optional = None) -> BoolTensor:
    """
    :param lengths: LongTensor (1-D)
    :param device:
    :return: BoolTensor
    """
    index = torch.arange(lengths.max(), device=device or lengths.device)
    return index.unsqueeze(0) < lengths.unsqueeze(1)

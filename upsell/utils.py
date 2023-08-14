import random
import torch
import numpy as np
from typing import Callable, Optional, List, Tuple, Union
from torch.utils.data import DataLoader
from torch import FloatTensor, BoolTensor


def set_rand_seed(seed, cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_eval_mode(module, root=True):
    if root:
        module.train()

    name = module.__class__.__name__
    if "Dropout" in name or "BatchNorm" in name:
        module.training = False
    for child_module in module.children():
        set_eval_mode(child_module, False)


def generate_sequence_mask(lengths, device=None):
    """
    :param lengths: LongTensor (1-D)
    :param device:
    :return: BoolTensor
    """
    index = torch.arange(lengths.max(), device=device or lengths.device)
    return index.unsqueeze(0) < lengths.unsqueeze(1)


def split_dataloader(data_loader: DataLoader, ratio: float):
    """
    Split a data loader into two data loaders, each with a portion of the dataset in the original
    data loader. Useful for splitting into train and validation data loaders.
    """
    dataset = data_loader.dataset
    n = len(dataset)
    lengths = [int(n * ratio), n - int(n * ratio)]
    datasets = torch.utils.data.random_split(dataset, lengths)

    copied_fields = ["batch_size", "num_workers", "collate_fn", "drop_last"]
    data_loaders = []
    for d in datasets:
        data_loaders.append(
            DataLoader(
                dataset=d, **{k: getattr(data_loader, k) for k in copied_fields}
            )
        )

    return tuple(data_loaders)


class KeyBucketedBatchSampler(torch.utils.data.Sampler):
    """
    Pseudo bucketed batch sampler.

    Sample in a way that puts similarly-sized records into the same batch.
    Args:
        keys: list of keys by which the same or nearby keys are allocated
          in the same or nearby batches.
        batch_size:
        drop_last: Whether to drop the last incomplete batch. Defaults to False.
        shuffle_same_key: Whether to shuffle the instances of the same keys. Defaults to False.
    """

    def __init__(self, keys: List[int], batch_size: int, drop_last: bool = False, shuffle_same_key: bool = True):
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


def convert_to_bucketed_dataloader(data_loader: DataLoader, key_fn: Optional[Callable] = None,
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


def get_freer_gpu(by="n_proc"):
    """
    Return the GPU index which has the largest available memory

    Returns:
        int: the index of selected GPU.
    """
    import os
    if os.environ.get("CUDA_DEVICE_ORDER", None) != "PCI_BUS_ID":
        raise RuntimeError(
            "Need CUDA_DEVICE_ORDER=PCI_BUS_ID to ensure " "consistent ID"
        )

    from pynvml import (
        nvmlInit,
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetComputeRunningProcesses,
        nvmlDeviceGetMemoryInfo,
    )

    nvmlInit()
    n_devices = nvmlDeviceGetCount()
    gpu_id, gpu_state = None, None
    for i in range(0, n_devices):
        handle = nvmlDeviceGetHandleByIndex(i)
        if by == "n_proc":
            temp = -len(nvmlDeviceGetComputeRunningProcesses(handle))
        elif by == "free_mem":
            temp = nvmlDeviceGetMemoryInfo(handle).free
        else:
            raise ValueError("`by` can only be 'n_proc', 'free_mem'.")
        if gpu_id is None or gpu_state < temp:
            gpu_id, gpu_state = i, temp

    return gpu_id


def batch_grad(
    func: Callable,
    inputs: FloatTensor,
    idx: Union[int, Tuple[int], List] = None,
    mask: BoolTensor = None,
) -> FloatTensor:
    """Compute gradients for a batch of inputs

    Args:
        func (Callable):
        inputs (FloatTensor): The first dimension corresponds the different
          instances.
        idx (Union[int, Tuple[int], List]): The index from the second dimension
          to the last. If a list is given, then the gradient of the sum of
          function values of these indices is computed for each instance.
        mask (BoolTensor):

    Returns:
        FloatTensor: The gradient for each input instance.
    """

    assert torch.is_tensor(inputs)
    assert (idx is None) != (
        mask is None
    ), "Either idx or mask (and only one of them) has to be provided."

    inputs.requires_grad_()
    out = func(inputs)

    if idx is not None:
        if not isinstance(idx, list):
            idx = [idx]

        indices = []
        for i in range(inputs.size(0)):
            for j in idx:
                j = (j,) if isinstance(j, int) else j
                indices.append((i,) + j)
        t = out[list(zip(*indices))].sum(-1)
    else:
        # [M, B, ...]
        out = out.view(-1, *mask.size())
        t = out.masked_select(mask).sum()

    gradients = torch.autograd.grad(t, inputs)[0]
    return gradients


def batch_integrated_gradient(
    func: Callable,
    inputs: FloatTensor,
    mask: BoolTensor = None,
    baselines: FloatTensor = None,
    steps: int = 50,
) -> FloatTensor:
    """
    Compute integrated gradient of an input with the given func

    Args:
        func (Callable): need to be able to run `func(inputs)`.
        inputs (FloatTensor): a batch of input instances. The first dimension
          corresponds to different instances in the batch.
        mask (BoolTensor): of the same shape as `func(inputs)`, and if given,
          `func(inputs)[mask].sum()` is used as the target function.
        baselines (FloatTensor, optional): When set to None, a zero baseline
        steps (int, optional): Defaults to 50.
    Returns:
        FloatTensor: batch of integrated gradient, one for each input instance.
          Should be of the same shape as `inputs`.
    """

    if baselines is None:
        baselines = torch.zeros_like(inputs)
    else:
        assert inputs.size() == baselines.size()
    batch_size = inputs.size(0)

    # scale inputs; size: (steps * batch_size, *)
    scaled_inputs = baselines.unsqueeze(dim=-1) + (
        inputs - baselines
    ).unsqueeze(dim=-1) * torch.linspace(0, 1, steps, device=inputs.device)
    scaled_inputs = (
        scaled_inputs.permute([-1, *range(inputs.ndimension())])
        .contiguous()
        .view(-1, *inputs.size()[1:])
    )

    grads = batch_grad(func, scaled_inputs, mask=mask)
    grads = grads.view(steps, batch_size, *grads.size()[1:])
    avg_grads = grads[1:].mean(dim=0)
    integrated_grads = (inputs - baselines) * avg_grads

    return integrated_grads

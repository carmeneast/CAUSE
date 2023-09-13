import random
import torch
import numpy as np


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

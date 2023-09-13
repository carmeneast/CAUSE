from typing import Callable, Union, Tuple, List

import torch
from torch import FloatTensor, BoolTensor


def batch_grad(
    func: Callable,
    inputs: FloatTensor,
    idx: Union[int, Tuple[int], List] = None,
    mask: BoolTensor = None,
) -> FloatTensor:
    """
    Compute gradients for a batch of inputs

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
    assert (idx is None) != (mask is None),\
        'Either idx or mask (and only one of them) must be provided.'

    inputs.requires_grad_()

    # Shape: [batch_size * steps, time_stamps-1, 1]
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
    elif mask is not None:
        # Sum cumulants from each step and timestamp
        out = out.view(-1, *mask.size())\
            .masked_select(mask)\
            .sum()
    else:
        raise ValueError('Either idx or mask must be provided.')

    # Compute the sum of gradients of outputs with respect to the inputs.
    # Shape: [batch_size * steps, time_stamps, n_event_types]
    gradients = torch.autograd.grad(out, inputs)[0]
    return gradients


def batch_integrated_gradient(
    func: Callable,
    inputs: FloatTensor,
    mask: BoolTensor = None,
    baselines: FloatTensor = None,
    steps: int = 50,
) -> FloatTensor:
    """
    Compute integrated gradient of an input with the given callable

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

    # Multiply event weights by a range of numbers between 0 and 1. Then stack in ascending order:
    # [[inputs * 0], [inputs * 1/(steps-1)], [inputs * 2/(steps-1)], ..., [inputs * 1]]
    # Shape: [batch_size * steps, time_steps, n_event_types+1]
    scaled_inputs = baselines.unsqueeze(dim=-1) +\
        (inputs - baselines).unsqueeze(dim=-1) * torch.linspace(0, 1, steps, device=inputs.device)
    scaled_inputs = (
        scaled_inputs.permute([-1, *range(inputs.ndimension())])
        .contiguous()
        .view(-1, *inputs.size()[1:])
    )

    # Get gradients for the inputs and the attribution target.
    # Shape: [batch_size * steps, time_stamps, n_event_types]
    grads = batch_grad(func, scaled_inputs, mask=mask)

    # Shape: [steps, batch_size, time_stamps, n_event_types]
    grads = grads.view(steps, batch_size, *grads.size()[1:])

    # Average gradients across all steps
    # Shape: [batch_size, time_stamps, n_event_types]
    avg_grads = grads[1:].mean(dim=0)

    # "Integrate" gradients over the difference between inputs and baseline
    # Shape: [batch_size, time_stamps, n_event_types]
    integrated_grads = (inputs - baselines) * avg_grads

    return integrated_grads

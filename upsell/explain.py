import torch
from torch import FloatTensor, BoolTensor
from typing import Callable


def batch_grad(function: Callable, inputs: FloatTensor, mask: BoolTensor) -> FloatTensor:
    """
    Compute gradients for a batch of inputs

    Args:
        function (Callable):
        inputs (FloatTensor):
        mask (BoolTensor):

    Returns:
        FloatTensor: The gradient for each input instance.
    """

    assert torch.is_tensor(inputs)
    inputs.requires_grad_()

    # Shape: [batch_size * steps, time_stamps-1, 1]
    out = function(inputs)

    # Sum cumulants from each account + timestamp at each step
    out_agg = out.view(-1, *mask.size())\
        .masked_select(mask)\
        .sum()

    # IF WE WERE GOING TO SUM THE CUMULANTS FOR ALL EVENT TYPES INSTEAD OF JUST ONE:
    # CURRENTLY THIS DOESN'T WORK BECAUSE AUTOGRAD IS RECORDING OPERATIONS FOR A SINGLE EVENT TYPE
    # For each event type, sum the cumulants from each account + timestep at each step
    # Shape: [n_event_types]
    # out_agg = out.view(-1, *mask.expand(*mask.size()[:-1], out.size()[-1]).size()) \
    #     .masked_select(mask.expand(*mask.size()[:-1], out.size()[-1])) \
    #     .view(-1, out.size()[-1]) \
    #     .sum(0)

    # Compute the sum of gradients of outputs with respect to the inputs.
    # Shape: [batch_size * steps, time_stamps, n_event_types]
    gradients = torch.autograd.grad(out_agg, inputs)[0]
    return gradients

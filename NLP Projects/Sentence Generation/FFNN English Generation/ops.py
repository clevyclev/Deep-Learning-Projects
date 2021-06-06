from itertools import accumulate

import numpy as np

from edugrad.ops import Operation, tensor_op

from hw3_ops import lookup_rows, multiply
from hw4_ops import softmax_rows, cross_entropy_loss


@tensor_op
class concat(Operation):
    """Takes a list of m [batch_size, N] tensors,
    and concatenates them `horizontally', producing one
    [batch_size, N * m] tensor.
    """

    @staticmethod
    def forward(ctx, *tensors: np.ndarray):
        ctx.extend([arr.shape[1] for arr in tensors])
        return np.concatenate(tensors, axis=1)

    @staticmethod
    def backward(ctx, grad_output):
        # throw away the last length, since splitting will go from (len-1:end)
        lengths = ctx[:-1]
        # get split_points by adding up the lengths
        # e.g. [100, 100] -> [100, 200]
        split_points = np.array(list(accumulate(lengths)))
        return np.split(grad_output, split_points, axis=1)


@tensor_op
class tanh(Operation):
    """ Compute forward/backward passes for
    tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    """
    @staticmethod
    def forward(ctx, a):
        # TODO: implement
        stuff = np.divide(np.exp(a) - np.exp(-a), np.exp(a) + np.exp(-a))
        ctx.append(stuff)
        return stuff

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: implement
        return [(1 - ctx[-1]**2) * grad_output]

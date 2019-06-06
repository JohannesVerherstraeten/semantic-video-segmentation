"""
TODO currently not used in project...
"""

import torch
import torch.nn


class SoftArgmax1D(torch.nn.Module):
    """
    Based on https://github.com/MWPainter/cvpr2019/blob/master/stitched/soft_argmax.py#L117

    See https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/ for more explanation.

    SoftArgMax1D(x) = \sum_i (i * softmax(x * \beta)_i)

    Beta is an arbitrary large number.
    """

    def __init__(self, beta=1000):
        super(SoftArgmax1D, self).__init__()
        self.beta = beta
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        smax = self.softmax(x * self.beta)
        smax = smax.transpose(1, 3).transpose(1, 2)
        indices = torch.arange(x.shape[1])
        return torch.matmul(smax, indices.float())

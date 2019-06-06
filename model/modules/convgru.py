"""
Copied from https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py

and refactored to look more like the torch GRU implementation

https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py

and like

https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable

from model.modules.baseconvrnn import BaseConvRNNCell, BaseConvRNN


class ConvGRUCell(BaseConvRNNCell):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_channels, hidden_channels, kernel_size, init_param=False, **kwargs):
        """
        Initialize ConvGRU cell.

        :param input_channels: Number of channels of input tensor.
        :type input_channels: int
        :param hidden_channels: Number of channels of hidden state.
        :type hidden_channels: int
        :param kernel_size: Size of the convolutional kernel.
        :type kernel_size: (int, int)
        """
        super().__init__(input_channels, hidden_channels, kernel_size)

        if init_param:
            raise NotImplementedError

        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.reset_gate = nn.Conv2d(in_channels=input_channels + hidden_channels,
                                    out_channels=hidden_channels,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding)
        self.update_gate = nn.Conv2d(in_channels=input_channels + hidden_channels,
                                     out_channels=hidden_channels,
                                     kernel_size=self.kernel_size,
                                     padding=self.padding)
        self.out_gate = nn.Conv2d(in_channels=input_channels + hidden_channels,
                                  out_channels=hidden_channels,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

        if len(kwargs) > 0:
            self.logger.error("Unknown kwargs given: {}".format(kwargs))

    def _forward(self, input_tensor, state: torch.Tensor):
        """
        Forward the input through the cell with the given state.

        :param input_tensor: Input for the cell. Shape: (batch_size, input_channels, height, width)
        :type input_tensor: torch.Tensor
        :param state: The state of the cell.
        :type state: torch.Tensor
        :return: The output of the cell, and the new state of the cell
        :rtype: torch.Tensor, (torch.Tensor | list[torch.Tensor])
        """
        stacked_inputs = torch.cat([input_tensor, state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_tensor, state * reset], dim=1)))
        new_state = state * (1 - update) + out_inputs * update

        # TODO return copy of state as result to avoid unwanted modification of state?
        return new_state, new_state

    def _init_new_state(self, input_tensor):
        """
        Create an initialization for the state, given the input.
        :param input_tensor: Input to create the new state for. Shape: (batch_size, input_channels, height, width)
        :type input_tensor: torch.Tensor
        :return: A new state.
        :rtype: Any
        """
        state_size = [input_tensor.shape[0], self.hidden_channels] + list(input_tensor.shape[2:])
        return Variable(input_tensor.new_zeros(state_size, requires_grad=True))


class ConvGRU(BaseConvRNN):
    """
    Generate a multi-layer convolutional GRU.

    E.g., setting num_layers=2 would mean stacking two GRUs together to form
    a `stacked GRU`, with the second GRU taking in outputs of the first GRU and computing the final
    results.

    Preserves spatial dimensions across cells, only altering depth.
    """

    def __init__(self, input_channels, hidden_channels, kernel_sizes, num_layers=1, init_param=False):
        """
        Initialize convolutional RNN module.

        :param input_channels: Number of channels of input tensor.
        :type input_channels: int
        :param hidden_channels: Number of channels of hidden states.
        :type hidden_channels: int | list[int]
        :param kernel_sizes: Size of the convolutional kernels of the (stacked) ConvRNN module.
        :type kernel_sizes: (int, int) | list[(int, int)]
        :param num_layers: Number of GRU layers.
        :type num_layers: int
        :param init_param: Whether to initialize the trainable parameters of this cell as a kind-of unit operation.
        :type init_param: bool
        """
        super(ConvGRU, self).__init__(input_channels, hidden_channels, kernel_sizes, num_layers, init_param=init_param)

    def _new_conv_rnn_cell(self, input_channels, hidden_channels, kernel_size, **kwargs):
        """
        Create a new convolutional RNN cell.

        :param input_channels: Number of channels of input tensor.
        :type input_channels: int
        :param hidden_channels: Number of channels of hidden state.
        :type hidden_channels: int
        :param kernel_size: Size of the convolutional kernel.
        :type kernel_size: (int, int)
        :return:
        :rtype: BaseConvRNNCell
        """
        return ConvGRUCell(input_channels, hidden_channels, kernel_size, **kwargs)

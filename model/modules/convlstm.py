"""
Copied from https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py

Modifications:
- comments added,
- slightly modified to be more similar to the torch implementation of the standard LSTM:

https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/rnn.py
"""
import torch
import torch.nn as nn
from torch.nn import init
import math

from torch.autograd import Variable

from model.modules.baseconvrnn import BaseConvRNN, BaseConvRNNCell


class ConvLSTMCell(BaseConvRNNCell):

    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True, init_param=False, **kwargs):
        """
        Initialize ConvLSTM cell.

        :param input_channels: Number of channels of input tensor.
        :type input_channels: int
        :param hidden_channels: Number of channels of hidden state.
        :type hidden_channels: int
        :param kernel_size: Size of the convolutional kernel.
        :type kernel_size: (int, int)
        :param bias: Whether or not to add the bias.
        :type bias: bool
        :param init_param: Whether to initialize the trainable parameters of this cell as a kind-of unit operation.
        :type init_param: bool
        """
        super(ConvLSTMCell, self).__init__(input_channels, hidden_channels, kernel_size)

        self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                              out_channels=4 * self.hidden_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        if init_param:
            self.reset_trainable_parameters()
        if len(kwargs) > 0:
            self.logger.error("Unknown kwargs given: {}".format(kwargs))

    def _forward(self, input_tensor, state):
        """
        Forward the input through the cell with the given state.

        :param input_tensor: Input for the cell. Shape: (batch_size, input_channels, height, width)
        :type input_tensor: torch.Tensor
        :param state: The state of the cell.
        :type state: Any
        :return: The output of the cell, and the new state of the cell
        :rtype: torch.Tensor, (torch.Tensor | list[torch.Tensor])
        """
        h_cur, c_cur = state
        combined = torch.cat((input_tensor, h_cur), dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)

    def _init_new_state(self, input_tensor):
        """
        Create an initialization for the state, given the input.
        :param input_tensor: Input to create the new state for. Shape: (batch_size, input_channels, height, width)
        :type input_tensor: torch.Tensor
        :return: A new state.
        :rtype: Any
        """
        batch_size = input_tensor.shape[0]
        spatial_size = input_tensor.shape[2:]
        # TODO: Must be wrapped in variable? -> I think yes
        # TODO: requires grad? -> I think yes
        state_size = [batch_size, self.hidden_channels] + list(spatial_size)
        h_cur = Variable(input_tensor.new_zeros(state_size, requires_grad=True))
        c_cur = Variable(input_tensor.new_zeros(state_size, requires_grad=True))

        return h_cur, c_cur

    def print_parameters(self):
        """
        For debugging purposes only.
        """
        # self.conv.weight.shape = [4*hidden_channels, in_+_hidden_channels, 3, 3]

        count_ig_it_ij = 0
        count_ig_it_ii = 0
        count_ig_st_ij = 0
        count_ig_st_ii = 0

        avg_ig_it_ii = torch.zeros(self.kernel_size).float()
        avg_ig_it_ij = torch.zeros(self.kernel_size).float()
        avg_ig_st_ii = torch.zeros(self.kernel_size).float()
        avg_ig_st_ij = torch.zeros(self.kernel_size).float()

        gate_nb = 3

        for i in range(self.conv.weight.shape[0]):
            if gate_nb * self.hidden_channels < i < (gate_nb+1) * self.hidden_channels:
                # input gate
                for j in range(self.conv.weight.shape[1]):
                    if j < self.input_channels:
                        # input tensor
                        if i % self.hidden_channels == j:
                            count_ig_it_ii += 1
                            avg_ig_it_ii += self.conv.weight[i, j]
                        else:
                            count_ig_it_ij += 1
                            avg_ig_it_ij += self.conv.weight[i, j]
                    else:
                        # state tensor
                        if i % self.hidden_channels + self.input_channels == j:
                            count_ig_st_ii += 1
                            avg_ig_st_ii += self.conv.weight[i, j]
                        else:
                            count_ig_st_ij += 1
                            avg_ig_st_ij += self.conv.weight[i, j]

        print("----------------------")
        print(self.conv.weight.shape)
        print("Hidden channels: " + str(self.hidden_channels))
        print("Input channels: " + str(self.input_channels))

        print(count_ig_it_ii)
        print(count_ig_it_ij)
        print(count_ig_st_ii)
        print(count_ig_st_ij)

        print(avg_ig_it_ii / count_ig_it_ii)
        print(avg_ig_it_ij / count_ig_it_ij)
        print(avg_ig_st_ii / count_ig_st_ii)
        print(avg_ig_st_ij / count_ig_st_ij)

    def reset_trainable_parameters(self):
        """
        TODO clean up
        """
        with torch.no_grad():
            # self.conv.weight.shape = [4*hidden_channels, in_+_hidden_channels, 3, 3]

            init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))     # Default pytorch init
            for i in range(self.conv.weight.shape[0]):
                if i < self.hidden_channels:
                    # input gate
                    for j in range(self.conv.weight.shape[1]):
                        # pass
                        if j < self.input_channels:
                            # input tensor
                            if i % self.hidden_channels == j:
                                # self.conv.weight[i, j, 1, 1] = 0.005
                                kernel = torch.zeros(self.kernel_size).float() + 0.001
                                kernel[1, 1] = 0.01
                                self.conv.weight[i, j] = kernel
                            # else:
                            #     kernel = torch.zeros(self.kernel_size).float() + 0.000000001
                            #     self.conv.weight[i, j] = kernel
                        else:
                            # state tensor
                            if i + self.input_channels == j:
                                # self.conv.weight[i, j, 1, 1] = 0.008
                                kernel = torch.zeros(self.kernel_size).float() + 0.001
                                kernel[1, 1] = 0.01
                                self.conv.weight[i, j] = kernel
                            # else:
                            #     kernel = torch.zeros(self.kernel_size).float() + 0.000000001
                            #     self.conv.weight[i, j] = kernel
                elif i < 2 * self.hidden_channels:
                    # forget gate
                    for j in range(self.conv.weight.shape[1]):
                        pass
                        if j < self.input_channels:
                            # input tensor
                            if i % self.hidden_channels == j:
                                # self.conv.weight[i, j, 1, 1] = 1.
                                kernel = torch.zeros(self.kernel_size).float() + 0.1
                                kernel[1, 1] = 0.6
                                self.conv.weight[i, j] = kernel
                            # else:
                            #     kernel = torch.zeros(self.kernel_size).float() + 0.000000001
                            #     self.conv.weight[i, j] = kernel
                        else:
                            # state tensor
                            if i + self.input_channels == j:
                                # self.conv.weight[i, j, 1, 1] = 1.
                                kernel = torch.zeros(self.kernel_size).float() + 0.1
                                kernel[1, 1] = 0.6
                                self.conv.weight[i, j] = kernel
                            # else:
                            #     kernel = torch.zeros(self.kernel_size).float() + 0.000000001
                            #     self.conv.weight[i, j] = kernel
                elif i < 3 * self.hidden_channels:
                    # output gate
                    for j in range(self.conv.weight.shape[1]):
                        if j < self.input_channels and i % self.hidden_channels == j:
                            # input tensor
                            # self.conv.weight[i, j, 1, 1] = 0.3
                            kernel = torch.zeros(self.kernel_size).float() + 0.001
                            kernel[1, 1] = 0.3
                            self.conv.weight[i, j] = kernel
                        # else:
                        #     # state tensor
                        #     kernel = torch.zeros(self.kernel_size).float() + 0.000000001
                        #     self.conv.weight[i, j] = kernel
                else:
                    # candidate new state
                    for j in range(self.conv.weight.shape[1]):
                        if j < self.input_channels and i % self.hidden_channels == j:
                            # input tensor
                            # self.conv.weight[i, j, 1, 1] = 0.3
                            kernel = torch.zeros(self.kernel_size).float() + 0.001
                            kernel[1, 1] = 0.3
                            self.conv.weight[i, j] = kernel
                        # else:
                        #     # state tensor
                        #     kernel = torch.zeros(self.kernel_size).float() + 0.000000001
                        #     self.conv.weight[i, j] = kernel


class ConvLSTM(BaseConvRNN):
    """
    Generate a multi-layer convolutional LSTM.

    E.g., setting num_layers=2 would mean stacking two LSTMs together to form
    a `stacked LSTM`, with the second LSTM taking in outputs of the first LSTM and computing the final
    results.

    Preserves spatial dimensions across cells, only altering depth.
    """

    def __init__(self, input_channels, hidden_channels, kernel_sizes, num_layers=1, init_param=False):
        """
        Initialize convolutional LSTM module.

        :param input_channels: Number of channels of input tensor.
        :type input_channels: int
        :param hidden_channels: Number of channels of hidden states.
        :type hidden_channels: int | list[int]
        :param kernel_sizes: Size of the convolutional kernels of the (stacked) ConvLSTM module.
        :type kernel_sizes: (int, int) | list[(int, int)]
        :param num_layers: Number of GRU layers.
        :type num_layers: int
        :param init_param: Whether to initialize the trainable parameters of this cell as a kind-of unit operation.
        :type init_param: bool
        """
        super(ConvLSTM, self).__init__(input_channels, hidden_channels, kernel_sizes, num_layers, init_param=init_param)

    def _new_conv_rnn_cell(self, input_channels, hidden_channels, kernel_size, **kwargs):
        """
        Create a new convolutional LSTM cell.

        :param input_channels: Number of channels of input tensor.
        :type input_channels: int
        :param hidden_channels: Number of channels of hidden state.
        :type hidden_channels: int
        :param kernel_size: Size of the convolutional kernel.
        :type kernel_size: (int, int)
        :return:
        :rtype: BaseConvRNNCell
        """
        return ConvLSTMCell(input_channels, hidden_channels, kernel_size, **kwargs)

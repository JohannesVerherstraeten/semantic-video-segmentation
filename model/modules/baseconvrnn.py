import torch
import torch.nn as nn
import logging

from torch.autograd import Variable


def _reduce_batch_size_if_needed(input_tensor, state):
    if isinstance(state, (list, tuple)):
        return [_reduce_batch_size_if_needed(input_tensor, state_item) for state_item in state]
    assert isinstance(state, torch.Tensor)

    if input_tensor.shape[0] < state.shape[0]:
        # Batch size is reduced. Discard the state of the last batches.
        return state[:input_tensor.shape[0], ...]
    elif input_tensor.shape[0] > state.shape[0]:
        raise RuntimeError("Batch size of the input is larger than that of the state.")
    else:
        return state


def _repackage(state):
    if isinstance(state, torch.Tensor):
        state.detach_()
    elif isinstance(state, (tuple, list)):
        [_repackage(state_elem) for state_elem in state]
    else:
        raise RuntimeError("Unexpected state type: got {}, expected (torch.Tensor | tuple | list)"
                           .format(type(state)))


class BaseConvRNNCell(nn.Module):
    """
    Base class for all convolutional RNN cells.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size):
        """
        Initialize convolutional RNN cell.

        :param input_channels: Number of channels of input tensor.
        :type input_channels: int
        :param hidden_channels: Number of channels of hidden state.
        :type hidden_channels: int
        :param kernel_size: Size of the convolutional kernel.
        :type kernel_size: (int, int)
        """
        super(BaseConvRNNCell, self).__init__()

        self.logger = logging.getLogger(self.__class__.__name__)

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self._state = None

    def forward(self, input_tensor):
        """
        Forward the input through the cell.

        :param input_tensor: Input for the cell. Shape: (batch_size, input_channels, height, width)
        :type input_tensor: torch.Tensor
        :return: The output of the cell.
        :rtype: torch.Tensor
        """
        self.check_forward_input(input_tensor)
        state = self._get_state(input_tensor)
        self.check_forward_state(input_tensor, state)

        result, new_state = self._forward(input_tensor, state)
        self._set_state(new_state)

        return result

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
        raise NotImplementedError

    def _init_new_state(self, input_tensor):
        """
        Create an initialization for the state, given the input.
        :param input_tensor: Input to create the new state for. Shape: (batch_size, input_channels, height, width)
        :type input_tensor: torch.Tensor
        :return: A new state.
        :rtype: Any
        """
        raise NotImplementedError

    def _get_state(self, input_tensor):
        """
        :param input_tensor:
        :type input_tensor: torch.Tensor
        :return:
        :rtype: A (nested) list of torch.Tensor elements of the form (batch_size, ...)
        """
        if self._state is None:
            return self._init_new_state(input_tensor)

        # If the batch size is reduced, discard the states of the last batches.

        return _reduce_batch_size_if_needed(input_tensor, self._state)

    def _set_state(self, state):
        self._state = state

    def reset_state(self):
        """
        Reset the internal state of this cell.
        """
        if self._state is not None:
            del self._state
            self._state = None

    def repackage_hidden_state(self):
        """
        Repackage the hidden state in new tensors to forget the gradient history.
        """

        if self._state is not None:
            _repackage(self._state)

    def reset_trainable_parameters(self):
        raise NotImplementedError

    def check_forward_input(self, input_tensor):
        """
        Check whether the given input is valid.

        :param input_tensor: shape: (batch_size, channels, height, width)
        :type input_tensor: torch.Tensor
        :return: an exception if the input is not valid.
        :rtype:
        """
        if input_tensor.shape[1] != self.input_channels:
            raise RuntimeError("Input has inconsistent input_channels: "
                               "got {}, expected {}.".format(input_tensor.shape[1], self.input_channels))

    def check_forward_state(self, input_tensor, state):
        """
        Check whether the given state is valid, given the input tensor.

        :param input_tensor: shape: (batch_size, input_channels, height, width)
        :type input_tensor: torch.Tensor
        :param state: (nested) list of Tensors with shape: (batch_size, input_channels, height, width)
        :type state: torch.Tensor | list[torch.Tensor]
        :return: an exception if the state is not valid given the input.
        :rtype:
        """
        if isinstance(state, (list, tuple)):
            [self.check_forward_state(input_tensor, state_elem) for state_elem in state]
            return

        if input_tensor.shape[0] != state.shape[0]:
            raise RuntimeError("Input batch size {} does not match hidden batch size {}"
                               .format(input_tensor.shape[0], state.shape[0]))
        if state.shape[1] != self.hidden_channels:
            raise RuntimeError("Hidden state has inconsistent hidden_channels: "
                               "got {}, expected {}".format(state.shape[1], self.hidden_channels))
        if input_tensor.shape[2] != state.shape[2] or input_tensor.shape[3] != state.shape[3]:
            raise RuntimeError("Input feature size {} does not match hidden feature size {}."
                               .format(input_tensor.shape[2:4], state.shape[2:4]))


class BaseConvRNN(nn.Module):
    """
    Base class for all convolutional RNN modules.

    E.g., setting num_layers=2 would mean stacking two ConvRNNs together to form
    a `stacked ConvRNN`, with the second ConvRNN taking in outputs of the first ConvRNN and computing the final
    results.

    Preserves spatial dimensions across cells, only altering depth.
    """
    def __init__(self, input_channels, hidden_channels, kernel_sizes, num_layers=1, **kwargs):
        """
        Initialize convolutional RNN module.

        :param input_channels: Number of channels of input tensor.
        :type input_channels: int
        :param hidden_channels: Number of channels of hidden states.
        :type hidden_channels: int | list[int]
        :param kernel_sizes: Size of the convolutional kernels of the (stacked) ConvRNN module.
        :type kernel_sizes: (int, int) | list[(int, int)]
        :param num_layers: Number of ConvRNN layers.
        :type num_layers: int
        """
        super(BaseConvRNN, self).__init__()

        self.logger = logging.getLogger(self.__class__.__name__)

        self.num_layers = num_layers
        self.input_channels = input_channels
        self.hidden_channels = self._extend_for_multilayer(hidden_channels)
        self.kernel_sizes = self._extend_for_multilayer(kernel_sizes)

        cell_list = []
        for layer in range(self.num_layers):
            input_dim = self.input_channels if layer == 0 else self.hidden_channels[layer - 1]

            cell = self._new_conv_rnn_cell(input_channels=input_dim,
                                           hidden_channels=self.hidden_channels[layer],
                                           kernel_size=self.kernel_sizes[layer],
                                           **kwargs)
            cell_list.append(cell)

        self.cells = nn.ModuleList(cell_list)

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
        raise NotImplementedError

    def _extend_for_multilayer(self, attribute):
        """
        Helper function, duplicating the given attribute `self.num_layers` times if it is a single value.

        If the attribute is a list, it checks whether the length is equal to `self.num_layers`.

        :param attribute:
        :type attribute: Any
        :return:
        :rtype: list[Any]
        """
        if isinstance(attribute, list):
            if len(attribute) == self.num_layers:
                return attribute
            else:
                raise RuntimeError("Number of attributes does not match number of layers: "
                                   "got {}, expected {} attributes.".format(attribute, self.num_layers))
        else:
            return [attribute] * self.num_layers

    def forward(self, input_tensor):
        """
        Forward the input through the cell.

        :param input_tensor: Input for the ConvRNN. Shape: (batch_size, input_channels, height, width)
        :type input_tensor: torch.Tensor
        :return: Output of the last layer of the ConvRNN. Shape: (batch_size, hidden_channels, height, width)
        :rtype: torch.Tensor
        """
        cur_layer_input = input_tensor
        cur_layer_output = None

        # propagate the input through each layer
        for layer in range(self.num_layers):
            cur_layer_output = self.cells[layer](input_tensor=cur_layer_input)

            # feed the output of the current layer as input to the next one
            cur_layer_input = cur_layer_output

        return cur_layer_output

    def reset_state(self):
        """
        Reset the internal state of this module.
        """
        for layer in range(self.num_layers):
            self.cells[layer].reset_state()

    def repackage_hidden_state(self):
        """
        Repackage the hidden state in new tensors to forget the gradient history.
        """
        for layer in range(self.num_layers):
            self.cells[layer].repackage_hidden_state()

    def reset_trainable_parameters(self):
        """
        Reset all trainable parameters.
        """
        for layer in range(self.num_layers):
            self.cells[layer].reset_trainable_parameters()

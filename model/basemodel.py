import logging
import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *args):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def reset_state(self):
        """
        Reset the internal state of this model.
        """
        pass

    def repackage_hidden_state(self):
        """
        Repackage the hidden state in new tensors to forget the gradient history.
        """
        pass

    def reset_trainable_parameters(self):
        """
        Reset all trainable parameters.
        """
        pass

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

    def recurrent_parameters(self):
        """
        An iterator over the parameters of the recurrent layers.
        """
        return iter([])


class BaseRNNModel(BaseModel):
    """
    Base class for all models containing recurrent layers.
    """
    def __init__(self):
        super(BaseRNNModel, self).__init__()

    def forward(self, *args):
        raise NotImplementedError

    def reset_state(self):
        raise NotImplementedError

    def repackage_hidden_state(self):
        raise NotImplementedError

    def recurrent_parameters(self):
        raise NotImplementedError

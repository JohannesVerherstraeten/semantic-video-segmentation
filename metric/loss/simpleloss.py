import torch
from torch.autograd import Variable
from torch import Tensor

from typing import *

from .baseloss import BaseLoss


class SimpleLoss(BaseLoss):

    def __init__(self, weight: float = 1.0, keep_predictions: bool = False, keep_labels: bool = False,
                 keep_frames: bool = False, timesteps: int = 0, name: str = None):
        """
        :param weight: If multiple losses are calculated, the resulting loss is the weighted sum of each individual loss.
        """
        super(SimpleLoss, self).__init__(keep_predictions, keep_labels, keep_frames, timesteps, name=name)
        self.weight = Variable(torch.tensor(weight))

        self.__value = None

    def _loss(self, predictions: Optional[Tensor] = None, labels: Optional[Tensor] = None, frames: Optional[Tensor] = None) -> Tuple[Optional[Tensor], Dict]:
        loss_result, loss_log = self._unweighted_loss(predictions, labels, frames)
        loss_log["value"] = None

        if loss_result is not None:
            loss_log["value"] = loss_result.detach().item()
            if self.__value is not None:
                self.__value += loss_result.detach().item()
            else:
                self.__value = loss_result.detach().item()
        final_loss_log = {self.name: loss_log}

        return self.__do_weighting(loss_result, final_loss_log)

    def _unweighted_loss(self, predictions: Optional[Tensor], labels: Optional[Tensor], frames: Optional[Tensor]) -> Tuple[Optional[Tensor], Dict]:
        raise NotImplementedError

    def value(self) -> Tuple[Optional[float], Dict]:
        log = {self.name: {"value": self.__value}}
        return self.__do_weighting(self.__value, log)

    def __do_weighting(self, loss_result, loss_log):
        """
        loss_result can be a Tensor or a float.
        """
        if loss_result is not None:
            loss_result *= self.weight
        loss_log[self.name]["weight"] = self.weight.item()

        return loss_result, loss_log

    def reset(self):
        super(SimpleLoss, self).reset()
        self.__value = None

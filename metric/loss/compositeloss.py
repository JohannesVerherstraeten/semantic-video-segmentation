from typing import *
from torch import Tensor

from .baseloss import BaseLoss
from .simpleloss import SimpleLoss


class CompositeLoss(BaseLoss):

    def __init__(self, *losses: SimpleLoss, name: str = None):
        super(CompositeLoss, self).__init__(name=name)

        total_weight = 0.
        for loss in losses:
            total_weight += loss.weight
        assert(total_weight - 1. < 0.0000001)

        self.losses = losses

    def _loss(self, predictions: Optional[Tensor] = None, labels: Optional[Tensor] = None, frames: Optional[Tensor] = None) -> Tuple[Optional[Tensor], Dict]:
        loss_result = None
        losses_log = {self.name: {"value": None, "sublosses": []}}
        for loss_function in self.losses:
            loss_result_i, loss_log_i = loss_function.loss(predictions=predictions,
                                                           labels=labels,
                                                           frames=frames)
            loss_result = self.__merge_losses(loss_result, loss_result_i)
            losses_log = self.__merge_logs(losses_log, loss_log_i)
        losses_log[self.name]["value"] = loss_result.detach().item() if loss_result is not None else None  # important to take the item() to prevent modificatino afterwards!

        return loss_result, losses_log

    def value(self) -> Tuple[Optional[float], Dict]:
        loss_value = None
        losses_log = {self.name: {"value": None, "sublosses": []}}
        for loss_function in self.losses:
            value, log = loss_function.value()
            loss_value = self.__merge_losses(loss_value, value)
            losses_log = self.__merge_logs(losses_log, log)
        losses_log[self.name]["value"] = loss_value

        return loss_value, losses_log

    def __merge_losses(self, final_loss, current_loss):
        if current_loss is not None:
            if final_loss is None:
                final_loss = current_loss
            else:
                final_loss += current_loss
        return final_loss

    def __merge_logs(self, final_log, current_log):
        final_log[self.name]["sublosses"].append(current_log)
        return final_log

    def reset(self):
        super(CompositeLoss, self).reset()
        for loss in self.losses:
            loss.reset()

    def reset_state(self):
        super(CompositeLoss, self).reset_state()
        for loss in self.losses:
            loss.reset_state()



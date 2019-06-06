import torch
from torch.autograd import Variable
from torch import Tensor

from typing import *

from ..basemetric import BaseMetric


class BaseLoss(BaseMetric):
    """
    A loss is a metric that is differentiable.
    """

    def __init__(self, keep_predictions: bool = False, keep_labels: bool = False, keep_frames: bool = False,
                 timesteps: int = 0, name: str = None):
        super(BaseLoss, self).__init__(keep_predictions, keep_labels, keep_frames, timesteps, name=name)

    def _add(self, predictions: Optional[Tensor] = None, labels: Optional[Tensor] = None, frames: Optional[Tensor] = None) -> Tuple[Optional[float], Dict]:
        loss_result, loss_log = self._loss(predictions, labels, frames)
        loss_result = loss_result.detach().item() if loss_result is not None else None
        return loss_result, loss_log

    def loss(self, predictions: Optional[Tensor] = None, labels: Optional[Tensor] = None, frames: Optional[Tensor] = None) -> Tuple[Optional[Tensor], Dict]:
        self._store_current_input(predictions, labels, frames)
        return self._loss(predictions, labels, frames)

    def _loss(self, predictions: Optional[Tensor] = None, labels: Optional[Tensor] = None, frames: Optional[Tensor] = None) -> Tuple[Optional[Tensor], Dict]:
        raise NotImplementedError

    def value(self) -> Tuple[Optional[float], Dict]:
        raise NotImplementedError

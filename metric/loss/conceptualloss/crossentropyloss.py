import torch.nn
from torch import Tensor

from typing import *

from ..simpleloss import SimpleLoss


class CrossEntropyLoss(SimpleLoss):
    """
    Wrapper class for torch.nn.CrossEntropyLoss
    """

    def __init__(self, weight: float, class_weights=None, name=None):
        super(CrossEntropyLoss, self).__init__(weight, name=name)
        self.torch_implementation = torch.nn.CrossEntropyLoss(weight=class_weights)

    def _unweighted_loss(self, predictions: Optional[Tensor] = None, labels: Optional[Tensor] = None, frames: Optional[Tensor] = None) -> Tuple[Optional[Tensor], Dict]:
        result = None
        if predictions is not None and labels is not None:
            result = self.torch_implementation(predictions, labels.detach())

        return result, dict()

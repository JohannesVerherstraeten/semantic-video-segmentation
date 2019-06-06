from torch import Tensor
from torch.autograd import Variable

from utils.opticalflow import *

from ..simpleloss import SimpleLoss


class ChangeLoss(SimpleLoss):
    """
    Loss that penalizes the differences between the consecutive predictions.
    """

    def __init__(self, weight: float, conceptual_loss: SimpleLoss, only_when_label: bool = False, name=None, **kwargs):
        super(ChangeLoss, self).__init__(weight, keep_predictions=True, timesteps=1, name=name)
        self.conceptual_loss = conceptual_loss
        self.only_when_label = only_when_label

    def _unweighted_loss(self, predictions: Optional[Tensor], labels: Optional[Tensor], frames: Optional[Tensor]) -> Tuple[Optional[Tensor], Dict]:
        final_log = {"subloss": dict()}

        if predictions is None:
            return None, final_log

        # For the time being, only compute the changeloss if the normal loss (crossentropy) is calculated too,
        # to avoid that the changeloss dominates the result.
        if self.only_when_label and labels is None:
            return None, final_log

        prev_predictions = self.get_previous_predictions()

        if prev_predictions is None:
            return None, final_log

        _, prev_predictions_max = torch.max(prev_predictions, 1)
        prev_predictions_max = Variable(prev_predictions_max)

        result, log = self.conceptual_loss.loss(predictions=predictions, labels=prev_predictions_max)
        final_log["subloss"] = log

        return result, final_log

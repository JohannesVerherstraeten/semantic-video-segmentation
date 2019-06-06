from torch import Tensor
from torch.autograd import Variable

from utils.opticalflow import *

from ..simpleloss import SimpleLoss


class WarpingLoss(SimpleLoss):
    """
    Loss BASED ON short-term loss of following paper, but NOT THE SAME:
    http://openaccess.thecvf.com/content_ECCV_2018/papers/Wei-Sheng_Lai_Real-Time_Blind_Video_ECCV_2018_paper.pdf

    For exact same loss, see WarpingLoss2.
    """

    def __init__(self, weight: float, conceptual_loss: SimpleLoss, only_when_label: bool = False, name=None, **kwargs):
        super(WarpingLoss, self).__init__(weight, keep_predictions=True, keep_frames=True, timesteps=1, name=name)
        self.conceptual_loss = conceptual_loss
        self.only_when_label = only_when_label

    def _unweighted_loss(self, predictions: Optional[Tensor], labels: Optional[Tensor], frames: Optional[Tensor]) -> Tuple[Optional[Tensor], Dict]:
        final_log = {"subloss": dict()}

        if predictions is None or frames is None:
            return None, final_log

        # For the time being, only compute the changeloss if the normal loss (crossentropy) is calculated too,
        # to avoid that the changeloss dominates the result.
        if self.only_when_label and labels is None:
            return None, final_log

        prev_frames = self.get_previous_frames()
        prev_predictions = self.get_previous_predictions()

        if prev_frames is None or prev_predictions is None:
            return None, final_log

        _, prev_predictions_max = torch.max(prev_predictions, 1)

        of = optical_flow(prev_frames, frames)
        prev_predictions_warped = warp_flow(prev_predictions_max, of)
        prev_predictions_warped = Variable(prev_predictions_warped)

        result, log = self.conceptual_loss.loss(predictions=predictions, labels=prev_predictions_warped)
        final_log["subloss"] = log

        return result, final_log


class WarpingLoss2(SimpleLoss):
    """
    Short-term loss of following paper:
    http://openaccess.thecvf.com/content_ECCV_2018/papers/Wei-Sheng_Lai_Real-Time_Blind_Video_ECCV_2018_paper.pdf
    """

    def __init__(self, weight: float, alpha: float = 50., only_when_label: bool = False, **kwargs):
        super(WarpingLoss2, self).__init__(weight, keep_predictions=True, keep_frames=True, timesteps=1)
        self.alpha = alpha
        self.only_when_label = only_when_label

    def _unweighted_loss(self, predictions: Optional[Tensor], labels: Optional[Tensor], frames: Optional[Tensor]) -> Tuple[Optional[Tensor], Dict]:
        if predictions is None or frames is None:
            return None, dict()

        # For the time being, only compute the changeloss if the normal loss (crossentropy) is calculated too,
        # to avoid that the changeloss dominates the result.
        if self.only_when_label and labels is None:
            return None, dict()

        prev_frames = self.get_previous_frames()
        prev_predictions = self.get_previous_predictions()

        if prev_frames is None or prev_predictions is None:
            return None, dict()

        of = optical_flow(prev_frames, frames)

        prev_predictions_warped, _ = warp_flow(prev_predictions, of)
        prev_predictions_warped = Variable(prev_predictions_warped)

        prev_frames_warped, mapping_confidence = warp_flow(prev_frames, of)

        visibility_mask = torch.exp(-self.alpha * torch.pow(torch.norm(frames - prev_frames_warped, dim=1, keepdim=False), 2))
        visibility_mask *= mapping_confidence
        visibility_mask = Variable(visibility_mask)

        result = visibility_mask * torch.norm(predictions - prev_predictions_warped, p=1, dim=1, keepdim=False)

        scale_factor = Variable(torch.tensor(1. / (frames.shape[-1] * frames.shape[-2])))
        if (frames.is_cuda):
            scale_factor = scale_factor.cuda()
        
        result = scale_factor * torch.sum(result, dim=(0, 1, 2))

        return result, dict()

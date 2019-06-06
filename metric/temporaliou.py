from typing import *
from torch.tensor import Tensor
import numpy as np

from .basemetric import BaseMetric
from .iou import IoU
from utils.opticalflow import *
from utils import transforms
from utils import visualize


class TemporalIoU(BaseMetric):

    def __init__(self, num_classes, ignore_indices=(), name: str = None, **kwargs):
        super(TemporalIoU, self).__init__(keep_predictions=True, keep_frames=True, timesteps=1, name=name)
        self.ignore_indices = ignore_indices
        ignore_indices_extended = tuple(ignore_indices) + (num_classes, )
        self.standard_iou = IoU(num_classes + 1, ignore_indices=ignore_indices_extended)
        self.num_classes = num_classes

    def _add(self, predictions: Optional[Tensor] = None, labels: Optional[Tensor] = None, frames: Optional[Tensor] = None) -> Tuple[Optional[float], Dict]:

        prev_predictions = self.get_previous_predictions()
        prev_frames = self.get_previous_frames()
        if predictions is None or frames is None or prev_predictions is None or prev_frames is None:
            return None, dict()

        if len(self.ignore_indices) > 0 and labels is None:
            return None, dict()

        _, prev_pred_label = torch.max(prev_predictions, 1)

        forward_flow, backward_flow = optical_flow_2(prev_frames, frames)

        prev_pred_label_warped, _ = warp_flow(prev_pred_label, backward_flow)

        # calculate the flow consistency in the reference frame of the current timestep.
        flow_consistency_map = forward_backward_consistency(forward_flow, backward_flow, ref_frame_a=False)

        # Visualization for debugging:
        # prev_img_cv2 = transforms.float_tensor_to_cv2_img(prev_frames[0])
        # curr_img_cv2 = transforms.float_tensor_to_cv2_img(frames[0])
        #
        # of_img = visualize.flow_to_bgr(backward_flow)
        #
        # # cv2.imshow("prev frame", prev_img_cv2)
        # cv2.imshow("curr frame", curr_img_cv2)
        # cv2.imshow("backward flow", of_img)
        # cv2.imshow("consistency map", flow_consistency_map.astype(np.uint8).transpose(1, 2, 0) * 255)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()

        for ignore_index in self.ignore_indices:
            prev_pred_label_warped[np.where(labels.cpu().numpy() == ignore_index)] = ignore_index

        # where the flow was incorrect, set the class to a value that will be ignored.
        prev_pred_label_warped[np.where(flow_consistency_map == 0)] = self.num_classes

        iou, iou_log = self.standard_iou.add(predictions=predictions, labels=prev_pred_label_warped)

        return iou, {self.name: {"value": iou, "IoU log": iou_log}}

    def value(self) -> Tuple[Optional[float], Dict]:
        """
        Returns the accumulated metric value of all previous input data.
        """
        iou, iou_log = self.standard_iou.value()

        return iou, {self.name: {"value": iou, "IoU log": iou_log}}

    def reset(self):
        """
        Resets the accumulated metric value of all input data and the metric state of the current sequence.

        Should happen at start of every epoch.
        """
        super(TemporalIoU, self).reset()
        self.standard_iou.reset()

    def reset_state(self):
        super(TemporalIoU, self).reset_state()
        self.standard_iou.reset_state()

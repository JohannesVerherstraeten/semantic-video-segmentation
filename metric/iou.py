"""
Heavily based on https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py
"""
from typing import *
from torch.tensor import Tensor
import numpy as np

from .basemetric import BaseMetric
from .confusionmetric import ConfusionMetric


class IoU(BaseMetric):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Args:
        num_classes (int): number of classes in the classification problem
        normalized (boolean, optional): Determines whether or not the confusion
            matrix is normalized or not. Default: False.
        ignore_indices (iterable, optional): Indices of the classes to ignore when
            computing the IoU. Can be an int, or any iterable of ints.
    """
    def __init__(self, num_classes, normalized=False, ignore_indices=(), name: str = None, **kwargs):
        super(IoU, self).__init__(name=name)

        self.conf_metric = ConfusionMetric(num_classes, normalized)
        self.ignore_indices = ignore_indices

    def _add(self, predictions: Optional[Tensor] = None, labels: Optional[Tensor] = None, frames: Optional[Tensor] = None) -> Tuple[Optional[float], Dict]:
        """Adds the predicted and target pair to the IoU metric.

        Args:
            predictions (Tensor): Can be a (N, K, H, W) tensor of
                predicted scores obtained from the model for N examples and K classes,
                or (N, H, W) tensor of integer values between 0 and K-1.
            labels (Tensor): Can be a (N, K, H, W) tensor of
                target scores for N examples and K classes, or (N, H, W) tensor of
                integer values between 0 and K-1.
        Returns the IoU of the current input. Internally, the IoU of all
        inputs is accumulated.
        """
        if predictions is None or labels is None:
            return None, dict()
        predicted = predictions
        target = labels

        # Dimensions check
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        current_conf, _ = self.conf_metric.add(predicted.view(-1), target.view(-1))
        mean_iou, iou = self.__confusion_matrix_to_iou(current_conf)
        return mean_iou, {self.name: {"value": mean_iou}}

    def value(self):
        conf_matrix, _ = self.conf_metric.value()
        mean_iou, iou = self.__confusion_matrix_to_iou(conf_matrix)
        return mean_iou, {self.name: {"value": mean_iou, "class IoU": iou, "conf_matrix": conf_matrix}}

    def reset(self):
        super(IoU, self).reset()
        self.conf_metric.reset()

    def __confusion_matrix_to_iou(self, conf_matrix: np.ndarray) -> Tuple[float, np.ndarray]:
        """Computes the IoU and mean IoU.

        The mean computation ignores NaN elements of the IoU array.

        Returns:
            Tuple: (mIoU, IoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        for i in self.ignore_indices:
            conf_matrix[:, i] = 0
            conf_matrix[i, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)

        return float(np.nanmean(iou)), iou

"""
TODO update this loss, make more efficient.
"""

import torch
import torch.nn.modules.loss
import torch.nn.functional as f
from torch.autograd import Variable
import cv2
import numpy as np
from typing import *

from ..simpleloss import SimpleLoss


class SmoothingLoss(SimpleLoss):

    def __init__(self, weight, kernel_size=(25, 51), sigma_y=10, sigma_x=20, only_when_label=False, name=None, **kwargs):
        """
        :param kernel_size: kernel size of the smoothing function. (height, width)
        :type kernel_size:
        :param sigma_y: all variation in height is in 99.7% of three standard deviations from center.
        :type sigma_y:
        :param sigma_x: all variation in width is in 99.7% of three standard deviations from center.
        :type sigma_x:
        """
        super(SmoothingLoss, self).__init__(weight, keep_predictions=True, timesteps=1, name=name)

        self.ksize = kernel_size
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.only_when_label = only_when_label

    def _unweighted_loss(self, predictions: Optional[torch.Tensor], labels: Optional[torch.Tensor], frames: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Dict]:
        """
        Predictions has shape (batch_size, nb_classes, height, width)
        """
        if predictions is None:
            return None, dict()

        # For the time being, only compute the changeloss if the normal loss (crossentropy) is calculated too,
        # to avoid that the changeloss dominates the result.
        if self.only_when_label and labels is None:
            return None, dict()

        prev_pred = f.softmax(self.get_previous_predictions(), dim=1)        # comment out for test examples below
        curr_pred = f.softmax(predictions, dim=1)              # comment out for test examples below

        kernel_size_cv2 = (self.ksize[1], self.ksize[0])

        prev_pred_np = prev_pred.cpu().numpy()
        prev_pred_blurred_np = np.empty_like(prev_pred_np)        # spatially blur the label probabilities
        for batch in range(prev_pred_np.shape[0]):
            for clss in range(prev_pred_np.shape[1]):
                prev_pred_blurred_np[batch, clss, ...] = cv2.GaussianBlur(prev_pred_np[batch, clss, ...],
                                                                          kernel_size_cv2, self.sigma_x, self.sigma_y)
        prev_pred_blurred = torch.from_numpy(prev_pred_blurred_np)
        if predictions.is_cuda:
            prev_pred_blurred = prev_pred_blurred.cuda()
        prev_pred_blurred = f.softmax(prev_pred_blurred, dim=1)   # comment out for test examples below
        prev_pred_blurred = Variable(prev_pred_blurred)

        result = torch.tensor(1.) / torch.mean(prev_pred_blurred * curr_pred)
        return result, dict()


# a = np.array(
#     [[[[0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0.],
#        [0., 0.2, 0.8, 0.5, 0., 0., 0.],
#        [0., 0.5, 0.8, 0.8, 0., 0., 0.],
#        [0., 0.2, 0.7, 0.4, 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0.]]]])
#
# b = np.array(
#     [[[[0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0.2, 0.8, 0.5, 0., 0.],
#        [0., 0., 0.5, 0.8, 0.8, 0., 0.],
#        [0., 0., 0.2, 0.7, 0.4, 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0.]]]])
#
# c = np.array(
#     [[[[0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0.2, 0.8, 0.5, 0.],
#        [0., 0., 0., 0.5, 0.8, 0.8, 0.],
#        [0., 0., 0., 0.2, 0.7, 0.4, 0.],
#        [0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0.]]]])
#
# d = np.array(
#     [[[[0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0.3, 0.8, 0.5, 0., 0.],
#        [0., 0., 0.8, 0.8, 0.8, 0., 0.],
#        [0., 0., 0.5, 0.7, 0.4, 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0.]]]])


# print(a.shape)
#
# lcl = SmoothingLoss(kernel_size=(3, 3), sigma_x=2, sigma_y=1)
# loss1 = lcl.add(torch.from_numpy(a), torch.from_numpy(a))
# loss2 = lcl.add(torch.from_numpy(a), torch.from_numpy(b))
# loss3 = lcl.add(torch.from_numpy(a), torch.from_numpy(c))
# loss4 = lcl.add(torch.from_numpy(a), torch.from_numpy(d))
# print(loss1)
# print(loss2)
# print(loss3)
# print(loss4)
#
# cel = CrossEntropyLoss()
# loss5 = cel(torch.from_numpy(a[0]), torch.from_numpy(a[0]))
# loss6 = CrossEntropyLoss(torch.from_numpy(a), torch.from_numpy(b))
# loss7 = CrossEntropyLoss(torch.from_numpy(a), torch.from_numpy(c))
# loss8 = CrossEntropyLoss(torch.from_numpy(a), torch.from_numpy(d))

# print(loss5)
# print(loss6)
# print(loss7)
# print(loss8)

# b = cv2.GaussianBlur(a, (3, 3), sigmaX=2, sigmaY=1)
# print(b)
# print(np.sum(b))
# cv2.imshow("a", a)
# cv2.imshow("b", b)
# cv2.waitKey(0)

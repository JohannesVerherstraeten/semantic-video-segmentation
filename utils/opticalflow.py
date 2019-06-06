import cv2
import numpy as np
import torch

from typing import *

from .transforms import *


def optical_flow(frame1: torch.Tensor, frame2: torch.Tensor, backward: bool = True) -> np.ndarray:
    """
    Calculate dense optical flow between two video frames.

    Supports mini-batches.

    Important: use backward optical flow for forward warping, and vice versa!

    See optical_flow_2 for calculation of both forward and backward optical flow.
    See util.transforms.py for explanation of image formats.

    :param frame1: video frame with shape (batch, channels, height, width) in float tensor format.
    :param frame2: video frame with shape (batch, channels, height, width) in float tensor format.
    :param backward: calculate backward (default) or forward optical flow)
    :return: optical flow in x- and y direction: (batch, height, width, 2) in cv2 format.
    """
    assert frame1.shape == frame2.shape

    # result shape (batch, height, width, 2)
    result = np.empty((frame1.size(0), frame1.size(2), frame1.size(3), 2))

    # treat each image in the batch separately
    for i in range(frame1.size(0)):

        frame1_i = float_tensor_to_cv2_img(frame1[0])
        frame1_i = cv2.cvtColor(frame1_i, cv2.COLOR_BGR2GRAY)

        frame2_i = float_tensor_to_cv2_img(frame2[0])
        frame2_i = cv2.cvtColor(frame2_i, cv2.COLOR_BGR2GRAY)

        if backward:
            flow = cv2.calcOpticalFlowFarneback(frame2_i, frame1_i, None, 0.5, 6, 15, 3, 5, 1.1, 0)
        else:
            flow = cv2.calcOpticalFlowFarneback(frame1_i, frame2_i, None, 0.5, 6, 15, 3, 5, 1.1, 0)

        result[i] = flow

    return result


def optical_flow_2(frame1: torch.Tensor, frame2: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the forward- and backward optical flow between the two frames.

    Supports mini-batches.

    Important: use backward optical flow for forward warping, and vice versa!

    :param frame1: video frame with shape (batch, channels, height, width) in float tensor format.
    :param frame2: video frame with shape (batch, channels, height, width) in float tensor format.
    :return: optical flow in x- and y direction: (batch, height, width, 2) in cv2 format.
    """
    assert frame1.shape == frame2.shape

    # result shape (batch, height, width, 2)
    result_forward = np.empty((frame1.size(0), frame1.size(2), frame1.size(3), 2))
    result_backward = np.empty((frame1.size(0), frame1.size(2), frame1.size(3), 2))

    # treat each image in the batch separately
    for i in range(frame1.size(0)):

        frame1_i = float_tensor_to_cv2_img(frame1[0])
        frame1_i = cv2.cvtColor(frame1_i, cv2.COLOR_BGR2GRAY)

        frame2_i = float_tensor_to_cv2_img(frame2[0])
        frame2_i = cv2.cvtColor(frame2_i, cv2.COLOR_BGR2GRAY)

        forward_flow = cv2.calcOpticalFlowFarneback(frame1_i, frame2_i, None, 0.5, 6, 15, 3, 5, 1.1, 0)
        backward_flow = cv2.calcOpticalFlowFarneback(frame2_i, frame1_i, None, 0.5, 6, 15, 3, 5, 1.1, 0)

        result_forward[i] = forward_flow
        result_backward[i] = backward_flow

    return result_forward, result_backward


def forward_backward_consistency(forward_flow: np.ndarray, backward_flow: np.ndarray, ref_frame_a: bool = True) -> np.ndarray:
    """
    Calculates forward-backward consistency.

    According to this paper: https://lmb.informatik.uni-freiburg.de/Publications/2010/Bro10e/sundaram_eccv10.pdf, p7.

    :param forward_flow: flow from frame a to frame b, with reference frame a.
    :param backward_flow: flow from frame b to frame a, with reference frame b.
    :param ref_frame_a: whether the consistency map is in reference frame a (default) or b.
    :return: Boolean map indicating where the forward and backward optical flow are consistent with each other.
        (batch, height, width).
    """
    assert forward_flow.shape == backward_flow.shape
    assert len(backward_flow.shape) == 4    # (batch, height, width, 2)

    if not ref_frame_a:
        backward_flow, forward_flow = forward_flow, backward_flow

    backward_flow_warped = np.transpose(backward_flow, axes=(0, 3, 1, 2))
    backward_flow_warped, _ = warp_flow_np(backward_flow_warped, forward_flow)
    backward_flow_warped *= -1.
    backward_flow_warped = np.transpose(backward_flow_warped, axes=(0, 2, 3, 1))

    flow_diff_norm = np.linalg.norm(forward_flow - backward_flow_warped, axis=3, keepdims=False) ** 2
    forward_flow_norm = np.linalg.norm(forward_flow, axis=3, keepdims=False) ** 2
    backward_flow_warped_norm = np.linalg.norm(backward_flow_warped, axis=3, keepdims=False) ** 2

    consistency_map = flow_diff_norm < 0.01 * (forward_flow_norm + backward_flow_warped_norm) + 0.5

    return consistency_map


def flow_to_pixel_map(flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a pixel map from the optical flow, and a corresponding confidence map.

    The pixel map gives for each pixel in the first image, the corresponding pixel in the second image,
    according to the given optical flow.

    The confidence map is a binary map, where a 0 indicates that the mapping will probably be wrong there,
    because of optical flow pointing from out of the image boundaries.

    :param flow: optical flow in x- and y direction: (batch, height, width, 2) in cv2 format.
    :return: mapping: (batch_size, height, width), confidence map: (batch_size, height, width)
    """
    h, w = flow.shape[1: 3]
    flow = flow.astype(np.float32)

    # ordinary mapping
    mapping = np.around(flow).astype(np.int)
    for i in range(mapping.shape[0]):
        mapping[i, ..., 0] += np.arange(w)
        mapping[i, ..., 1] += np.arange(h)[:, np.newaxis]

    # where will the mapping be out of range?
    x_zero_bound = np.where(mapping[..., 0] < 0)
    y_zero_bound = np.where(mapping[..., 1] < 0)
    x_max_bound = np.where(mapping[..., 0] >= w)
    y_max_bound = np.where(mapping[..., 1] >= h)

    # create confidence mask:
    mapping_confidence = np.ones(mapping.shape[:3], dtype=np.float)
    mapping_confidence[x_zero_bound] = 0.
    mapping_confidence[y_zero_bound] = 0.
    mapping_confidence[x_max_bound] = 0.
    mapping_confidence[y_max_bound] = 0.

    # prevent mapping from out of range
    mapping[x_zero_bound] = 0
    mapping[y_zero_bound] = 0
    too_wide = (*x_max_bound, np.repeat(0, len(x_max_bound[0])))
    mapping[too_wide] = w-1
    too_high = (*y_max_bound, np.repeat(1, len(y_max_bound[0])))
    mapping[too_high] = h-1

    return mapping, mapping_confidence


def warp_flow(tensor: torch.Tensor, flow: np.ndarray) -> (torch.Tensor, torch.Tensor):
    """
    Warp the tensor according to the flow.

    :param tensor: tensor with shape (batch, channels, height, width) in float tensor format
        or with shape (batch, height, width) in long tensor format.
    :param flow: optical flow in x- and y direction: (batch, height, width, 2) in cv2 format.
    :return: warped tensor with shape (batch, channels, height, width) in float tensor format.
    """
    tensor_np = tensor.cpu().numpy()
    result, mapping_confidence = warp_flow_np(tensor_np, flow)

    result = torch.as_tensor(result, dtype=tensor.dtype)
    mapping_confidence = torch.as_tensor(mapping_confidence, dtype=tensor.dtype)
    if tensor.is_cuda:
        result = result.cuda()
        mapping_confidence = mapping_confidence.cuda()

    return result, mapping_confidence


def warp_flow_np(tensor: np.ndarray, flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp the tensor according to the flow.

    :param tensor: np.array with shape (batch, channels, height, width) in float tensor format
        or with shape (batch, height, width) in long tensor format.
    :param flow: optical flow in x- and y direction: (batch, height, width, 2) in cv2 format.
    :return: warped np.array with shape (batch, channels, height, width) in float tensor format.
    """
    assert tensor.shape[0] == flow.shape[0]
    assert tensor.shape[2:] == flow.shape[1:3] or tensor.shape[1:] == flow.shape[1:3]

    mapping, mapping_confidence = flow_to_pixel_map(flow)

    # TODO should be possible to do this cleaner using proper numpy indexing
    result = np.empty_like(tensor)
    for i in range(tensor.shape[0]):
        if len(tensor.shape) == 4:
            for j in range(tensor.shape[1]):
                result[i, j] = tensor[i, j, mapping[i, ..., 1], mapping[i, ..., 0]]      # cv2 x and y positions
        else:
            result[i] = tensor[i, mapping[i, ..., 1], mapping[i, ..., 0]]  # cv2 x and y positions

    return result, mapping_confidence


# def warp_flow(img: torch.Tensor, flow: np.ndarray) -> torch.Tensor:
#     """
#     Warp the image according to the flow.
#
#     :param img: image with shape (batch, channels, height, width) in float tensor format
#         or with shape (batch, height, width) in long tensor format.
#     :param flow: optical flow in x- and y direction: (batch, height, width, 2) in cv2 format.
#     :return: warped image with shape (batch, channels, height, width) in float tensor format.
#     """
#     assert img.size(0) == flow.shape[0]
#     assert img.shape[2:] == flow.shape[1:3] or img.shape[1:] == flow.shape[1:3]
#
#     if img.dtype == torch.long:
#         img_cv2 = (img.cpu().numpy()).astype(np.uint8)    # TODO: long_tensor_to_cv2: no batched single-channel conversion available
#     else:
#         img_cv2 = float_tensor_to_cv2_img(img)
#
#     h, w = flow.shape[1:3]
#     flow = flow.astype(np.float32)
#
#     # result shape (batch, channels, height, width)
#     result_np = np.empty(img_cv2.shape)
#
#     # treat each image in the batch separately
#     for i in range(img.size(0)):
#
#         flow[i, :, :, 0] += np.arange(w)
#         flow[i, :, :, 1] += np.arange(h)[:, np.newaxis]
#
#         res_i = cv2.remap(img_cv2[i], flow[i], None, cv2.INTER_NEAREST)   # TODO INTER_LINEAR?
#
#         # if len(res.shape) == len(img_cv2.shape) - 1:
#         #     res = res[np.newaxis, :]
#         # else:
#         #     res = np.transpose(res, (2, 0, 1))
#
#         result_np[i] = res_i
#
#     if img.dtype == torch.long:
#         img_tensor = torch.from_numpy(result_np).long()
#     else:
#         img_tensor = cv2_img_to_float_tensor(result_np)
#
#     if img.is_cuda:
#         img_tensor = img_tensor.cuda()
#
#     return img_tensor

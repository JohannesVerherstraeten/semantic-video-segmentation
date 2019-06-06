"""
Transform utilities.

Images:
- generally, this project assumes batched images. Batched images are stacked along the 0th
  dimension.
- image formats used in this project:
    - cv2 format: (height, width, channels) in BGR with values in range 0..255.
    - float tensor format: (channels, height, width) in RGB with values in range 0..1.
    - long tensor format: (channels, height, width) in RGB with natural numbers.

TODO: use tensor_to_cv2_img() and differentiate on float/long type with img.dtype == torch.long?

TODO: Switch case on shape is not clean.

TODO: use (single/double)-dispatch software design pattern? with separate classes for each type of image?
"""

import cv2
import numpy as np
import torch


def seconds_to_hms(seconds):
    """
    Convert seconds to hours, minutes, seconds.
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


def remap(image, old_values, new_values):
    """
    Remap the values of the input image. Useful for labels.

    https://github.com/davidtvs/PyTorch-ENet/blob/master/data/utils.py
    """
    assert isinstance(image, np.ndarray), "image must be of type PIL.Image or numpy.ndarray"
    assert type(new_values) is tuple, "new_values must be of type tuple"
    assert type(old_values) is tuple, "old_values must be of type tuple"
    assert len(new_values) == len(old_values), "new_values and old_values must have the same length"

    # Replace old values by the new ones
    result = np.zeros_like(image)
    for old, new in zip(old_values, new_values):
        # Since tmp is already initialized as zeros we can skip new values equal to 0
        if new != 0:
            result[image == old] = new
    return result


def long_tensor_to_cv2_img(tensor):
    """
    Convert LongTensor to image in cv2 format.

    TODO Switch case on shape is not clean.

    - cv2 format: (height, width, channels) in BGR with values in range 0..255.
    - long tensor format: (channels, height, width) in RGB with natural numbers.

    :param tensor: (batches, channels, height, width) or (channels, height, width), dtype=long
    :return: (batches, height, width, channels) or (height, width, channels), dtype=uint8

    :type tensor: torch.Tensor
    :rtype: np.ndarray
    """
    result = (tensor.detach().cpu().numpy()).astype(np.uint8)
    if len(result.shape) == 4:      # batched, multi-channel
        result = np.moveaxis(result, 1, -1)
        result = result[..., ::-1]
    elif len(result.shape) == 3:    # not batched, multi-channel
        result = np.moveaxis(result, 0, -1)
        result = result[..., ::-1]
    elif len(result.shape) == 2:    # not batched, single-channel
        result = result
    else:
        raise RuntimeError("Unexpected tensor dimensions: expected (2 | 3 | 4), got {}".format(len(tensor.shape)))

    return result


def float_tensor_to_cv2_img(tensor):
    """
    Convert Tensor to image in cv2 format.

    - cv2 format: (height, width, channels) in BGR with values in range 0..255.
    - float tensor format: (channels, height, width) in RGB with values in range 0..1.

    :param tensor: (batches, channels, height, width) or (channels, height, width), dtype=float
    :return: (batches, height, width, channels) or (height, width, channels), dtype=uint8

    :type tensor: torch.Tensor
    :rtype: np.ndarray
    """
    return long_tensor_to_cv2_img(tensor.detach() * 255)


def cv2_img_to_long_tensor(img):
    """
    Convert image in cv2 format to LongTensor.

    TODO Switch case on shape is not clean.

    - cv2 format: (height, width, channels) in BGR with values in range 0..255.
    - long tensor format: (channels, height, width) in RGB with natural numbers.

    :param img: (batches, height, width, channels) or (height, width, channels) or (height, width), dtype=uint8
    :return: (batches, channels, height, width) or (channels, height, width) or (height, width), dtype=long

    :type img: np.ndarray
    :rtype: torch.Tensor
    """
    if len(img.shape) == 4:     # batched, multi-channel
        img_rgb = np.empty_like(img)
        img_rgb[..., ::1] = img[..., ::-1]
        result = np.moveaxis(img_rgb, -1, 1)
    elif len(img.shape) == 3:   # not batched, multi-channel
        img_rgb = np.empty_like(img)
        img_rgb[..., ::1] = img[..., ::-1]
        result = np.moveaxis(img_rgb, -1, 0)
    elif len(img.shape) == 2:   # not batched, single-channel
        result = img
    else:
        raise RuntimeError("Unexpected image dimensions: expected (2 | 3 | 4), got {}".format(len(img.shape)))
    result = torch.from_numpy(result).long()
    return result


def cv2_img_to_float_tensor(img):
    """
    Convert image in cv2 format to Tensor.

    - cv2 format: (height, width, channels) in BGR with values in range 0..255.
    - float tensor format: (channels, height, width) in RGB with values in range 0..1.

    :param img: (batches, height, width, channels) or (height, width, channels), dtype=uint8
    :return: (batches, channels, height, width) or (channels, height, width), dtype=float

    :type img: np.ndarray
    :rtype: torch.Tensor
    """
    return cv2_img_to_long_tensor(img).float() / 255.

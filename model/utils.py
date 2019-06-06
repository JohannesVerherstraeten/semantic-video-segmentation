# TODO adapt to current framework! Do not use.

import cv2
import numpy as np
import torch


def optical_flow(frame1, frame2, flow_init=None):
    """
    Calculate dense optical flow between two frames.

    Does not support minibatches.

    :type frame1: torch.Tensor
    :type frame2: torch.Tensor
    :type flow_init: Optional[np.ndarray]
    :rtype: np.ndarray
    """
    frame1 = (np.transpose(frame1.numpy()[0], (1, 2, 0)) * 255).astype(np.uint8)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    frame2 = (np.transpose(frame2.numpy()[0], (1, 2, 0)) * 255).astype(np.uint8)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 6, 15, 3, 5, 1.1, 0)
    # flow = cv2.calcOpticalFlowFarneback(frame1, frame2, flow_init, 0.5, 6, 15, 3, 5, 1.1, 0)

    return flow


def optical_flow_batched(prev_frame, frame_batch, flow_init=None):
    """
    Calculate dense optical flow between the frames in the given mini-batch.

    Assuming the input is the last frame of a batch and the next batch of three frames.
    | a3 |        | b1 | b2 | b3 |,
    the calculated optical flow is:
    | a3->b1 | b1->b2 | b2->b3 |.

    If prev_frame is None, only
    | b1->b2 | b2->b3 |
    is returned.

    :param prev_frame: greyscale image in cv2 format: (360, 480) depth=uint8
    :param frame_batch: batch of greyscale images in cv2 format: (batches, 360, 480) depth=uint8
    :return: flow in x- and y direction: (batches, 360, 480, 2)

    :type prev_frame: np.ndarray
    :type frame_batch: np.ndarray
    :type flow_init: np.ndarray
    :rtype: np.ndarray
    """
    # Expected input: (batches, height, width)
    assert len(frame_batch.shape) == 3
    assert prev_frame is None or len(prev_frame.shape) == 2
    assert prev_frame is None or prev_frame.shape == frame_batch.shape[1:]

    if prev_frame is None and frame_batch.shape[0] == 1:
        return None

    # don't use flow initialization.
    flow_init = None

    # Result if prev_frame is None; (batches-1, height, width, 2)
    # Otherwise: (batches, height, width, 2)
    one_less = prev_frame is None
    result = np.empty((frame_batch.shape[0] - one_less, *frame_batch.shape[1:3], 2))

    if not one_less:
        result[0] = cv2.calcOpticalFlowFarneback(prev_frame, frame_batch[0], flow_init, 0.5, 6, 15, 3, 5, 1.1, 0)

    for i in range(1, frame_batch.shape[0]):
        result[i-one_less] = cv2.calcOpticalFlowFarneback(frame_batch[i-1], frame_batch[i], flow_init, 0.5, 6, 15, 3,
                                                          5, 1.1, 0)

    return result


def warp_flow(img, flow):
    """
    Warp image according to the flow.

    Does not support minibatches.

    :type img: torch.Tensor
    :type flow: np.ndarray
    :rtype: torch.Tensor
    """
    if len(img.data.shape) == 4:
        batched = True
        img = img[0]
    else:
        batched = False

    if img.dtype == torch.long:
        img_np = (img.numpy()).astype(np.uint8)
    else:
        img_np = (img.numpy() * 255).astype(np.uint8)

    img_np = np.transpose(img_np, (1, 2, 0))

    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(img_np, flow, None, cv2.INTER_NEAREST)

    if len(res.shape) == len(img_np.shape) - 1:
        res = res[np.newaxis, :]
    else:
        res = np.transpose(res, (2, 0, 1))

    if img.dtype == torch.long:
        res = torch.from_numpy(res).long()
    else:
        res = torch.from_numpy(res/255.).float()

    if batched:
        res = res.reshape((1, *res.data.shape))

    return res


def warp_flow_batched(prev_frame, frame_batch, flow_batch):
    """
    Warp the given frames according to the corresponding optical flow.

    Assuming the image input is a batch of three frames
    | a3 |          | b1 | b2 | b3 |,
    the flow must be
    | a3->b1 | b1->b2 | b2->b3 |.

    If prev_frame is None, the flow must be
    | b1->b2 | b2->b3 |.
    In this case, a batch with one less frame is returned.

    :param prev_frame: image in cv2 format: (360, 480, channels) depth=uint8
    :param frame_batch: batch of images in cv2 format: (batches, 360, 480, channels) depth=uint8
    :param flow_batch: flow in x- and y direction: (batches, 360, 480, 2)
    :return all frames except for the last one are warped according to their corresponding flow.

    :type prev_frame: np.ndarray
    :type frame_batch: np.ndarray
    :type flow_batch: np.ndarray
    :rtype: np.ndarray
    """
    # Expected frame input: (height, width, channels)
    # Expected frame batch input: (batch, height, width, channels)
    # Expected flow batch input: (batch, height, width, 2)
    assert prev_frame is None or len(prev_frame.shape) == 3
    assert len(frame_batch.shape) == 4
    assert flow_batch is None or len(flow_batch) == 4
    assert (prev_frame is None) or (prev_frame.shape[:2] == frame_batch.shape[1:3] and
                                    frame_batch.shape[1:3] == flow_batch[1:3])
    assert (prev_frame is None) == (flow_batch is None)

    if (prev_frame is None) and (flow_batch is None):
        return None

    if prev_frame is not None:
        assert prev_frame.shape == frame_batch.shape[1:]
        assert frame_batch.shape[0] == flow_batch.shape[0]
    else:
        assert flow_batch.shape[0] == frame_batch.shape[0] - 1

    one_less = prev_frame is None
    result = np.empty((frame_batch.shape[0] - one_less, *frame_batch.shape[1:]), dtype=frame_batch.dtype)

    if prev_frame is not None:
        result[0] = __warp_flow(prev_frame, flow_batch[0])

    for i in range(frame_batch.shape[0] - 1):
        result[i+1-one_less] = __warp_flow(frame_batch[i], flow_batch[i+1-one_less])

    return result


def __warp_flow(frame, flow):
    h, w = frame.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]

    return cv2.remap(frame, flow, None, cv2.INTER_NEAREST)

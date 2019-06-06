"""
TODO avoid duplicate code. Create classes for a visualizer to avoid continuously passing the fig and ax params.
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
import time
import gc
import logging
from typing import *

import utils.datautils as datautils
import utils.transforms as transforms
import utils.opticalflow as opticalflow

logger = logging.getLogger("Visualizer")


def batch_transform(batch, transform):
    """Applies a transform to a batch of samples.

    Keyword arguments:
    - batch (): a batch of samples
    - transform (callable): A function/transform to apply to ``batch``

    """
    # Convert the single channel label to RGB in tensor form
    # 1. torch.unbind removes the 0-dimension of "labels" and returns a tuple of
    # all slices along that dimension
    # 2. the transform is applied to each slice
    transf_slices = []
    for tensor in torch.unbind(batch):
        new_tensor = tensor
        if isinstance(transform, (tuple, list)):
            for transf in transform:
                new_tensor = transf(new_tensor)
        else:
            new_tensor = transform(new_tensor)
        transf_slices.append(new_tensor)

    return torch.stack(transf_slices)


def _cmap(img_cv2, color_encoding):
    """
    Quick color mapping function
    """
    result = np.empty((*img_cv2.shape, 3))
    for i, (class_name, color) in enumerate(color_encoding.items()):
        result[np.where(img_cv2 == i)] = (np.array(color) * 255).astype(np.uint8)
    return result


class Visualizer(object):

    def __init__(self, models=[], show_flow=False, overlay=True, window_size=(15, 7), show_labels=True):
        super(Visualizer, self).__init__()

        self.captions = [model.__class__.__name__ for model in models]

        self.nb_subplots_v = 1 + (1 - overlay) * show_labels + show_flow * 4
        self.nb_subplots_h = max(1, len(models))

        self.overlay = overlay
        self.window_size = window_size
        self.show_labels = show_labels

        self.fig, subplots = plt.subplots(self.nb_subplots_v, self.nb_subplots_h, figsize=self.window_size)
        self.subplots = np.array(subplots) if isinstance(subplots, (list, tuple, np.ndarray)) else np.array([subplots])
        if len(self.subplots.shape) > 1:
            self.subplots = self.subplots.flatten()
        self.subplotImgs = [None] * len(self.subplots)
        for i, caption in enumerate(self.captions):
            self.subplots[i].set_title(caption)

    def imshow(self, images, labels=None, class_encoding=None, legend=True, save_path=None):
        assert isinstance(images, torch.Tensor)
        images = torchvision.utils.make_grid(images)
        images_cv2 = transforms.float_tensor_to_cv2_img(images)

        if isinstance(labels, torch.Tensor):
            labels = [labels]

        outputs = [None] * self.nb_subplots_h * self.nb_subplots_v

        contains_none = False
        for label in labels:
            if label is None:
                contains_none = True
        if self.show_labels and labels is not None and len(labels) > 0 and not contains_none:
            assert isinstance(labels, list) and (len(labels) == 0 or all(isinstance(label, torch.Tensor) for label in labels))
            assert class_encoding is not None
            assert len(labels) == self.nb_subplots_h

            color_predictions = []
            for label in labels:
                # TODO optimize
                label_rgb_tensor = batch_transform(label.cpu(), [transforms.long_tensor_to_cv2_img,
                                                                 lambda x: _cmap(x, class_encoding),
                                                                 transforms.cv2_img_to_float_tensor])
                color_prediction = torchvision.utils.make_grid(label_rgb_tensor)
                color_prediction = transforms.float_tensor_to_cv2_img(color_prediction)
                color_predictions.append(color_prediction)

            if self.overlay:
                for i, color_prediction in enumerate(color_predictions):
                    output = (0.6 * images_cv2) + (0.4 * color_prediction)
                    output = output.astype(np.uint8)
                    outputs[i] = output
            else:
                for i, color_prediction in enumerate(color_predictions):
                    outputs[i] = images_cv2
                    outputs[i+self.nb_subplots_h] = color_prediction
        else:
            for i in range(self.nb_subplots_h):
                outputs[i] = images_cv2

        self.__imshow(outputs, class_encoding, legend, save_path)

    def __imshow(self, images, class_encoding=None, legend=True, save_path=None):
        """
        Internal function to show a grid of images.

        :param images: list of images in cv2 format.
        :type images: [np.ndarray]
        :param class_encoding:
        :type class_encoding:
        :param legend:
        :type legend:
        :param save_path:
        :type save_path:
        :return:
        :rtype:
        """
        assert isinstance(images, (list, tuple, np.ndarray))
        assert len(images) > 0
        if isinstance(images, np.ndarray):
            assert images.dtype == np.uint8
            assert images.shape[0] == self.nb_subplots_h * self.nb_subplots_v
        else:
            for image in images:
                assert image is not None
                assert isinstance(image, np.ndarray)
                assert image.dtype == np.uint8
            assert len(images) == self.nb_subplots_h * self.nb_subplots_v

        for i, image in enumerate(images):
            if self.subplotImgs[i] is None:
                self.subplotImgs[i] = self.subplots[i].imshow(image[..., ::-1])     # BGR to RGB
            else:
                self.subplotImgs[i].set_data(image[..., ::-1])      # BGR to RGB

        if legend:
            patches = []
            for class_name, color in class_encoding.items():
                patches.append(mpatches.Patch(color=color[::-1], label=class_name))
            plt.legend(handles=patches, bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)

        if save_path is not None:
            self.fig.savefig(save_path)

    def imshow_flow(self, images1, images2, flow_bw, flow_fw, warped, warp_error, flow_consistency_map):

        assert isinstance(images1, torch.Tensor)
        images1 = torchvision.utils.make_grid(images1)
        images1_cv2 = transforms.float_tensor_to_cv2_img(images1)

        assert isinstance(images2, torch.Tensor)
        images2 = torchvision.utils.make_grid(images2)
        images2_cv2 = transforms.float_tensor_to_cv2_img(images2)

        flow_bw_cv2 = flow_to_bgr(flow_bw)
        flow_fw_cv2 = flow_to_bgr(flow_fw)

        assert isinstance(warped, torch.Tensor)
        warped = torchvision.utils.make_grid(warped)
        warped_cv2 = transforms.float_tensor_to_cv2_img(warped)

        assert isinstance(warp_error, torch.Tensor)
        warp_error = torchvision.utils.make_grid(warp_error)
        warp_error_cv2 = transforms.float_tensor_to_cv2_img(warp_error)

        assert isinstance(flow_consistency_map, np.ndarray)
        flow_consistency_map_cv2 = flow_consistency_map.astype(np.uint8).transpose(1, 2, 0) * 255
        flow_consistency_map_cv2 = flow_consistency_map_cv2.repeat(3, axis=2)

        # # debugging:
        # fig, subplots = plt.subplots(3, 2, figsize=(7, 10))
        # subplots[0][0].imshow(images1_cv2[..., ::-1])
        # subplots[0][0].set_title("t-1")
        # subplots[0][1].imshow(images2_cv2[..., ::-1])
        # subplots[0][1].set_title("t")
        # subplots[1][0].imshow(flow_fw_cv2[..., ::-1])
        # subplots[1][0].set_title("forward flow")
        # subplots[1][1].imshow(flow_bw_cv2[..., ::-1])
        # subplots[1][1].set_title("backward flow")
        # subplots[2][0].imshow(warp_error_cv2[..., ::-1])
        # subplots[2][0].set_title("warping error")
        # subplots[2][1].imshow(flow_consistency_map_cv2[..., ::-1])
        # subplots[2][1].set_title("flow consistency")
        # plt.show()

        outputs = np.array([images1_cv2, images2_cv2, flow_fw_cv2, flow_bw_cv2, flow_consistency_map_cv2])

        outputs = outputs.repeat(self.nb_subplots_h, axis=0)
        self.__imshow(outputs, legend=False)


def flow_to_bgr(of):
    assert of.shape[0] == 1
    flow = of[0]
    h, w = flow.shape[:2]
    hsv = np.empty((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / (np.pi * 2)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


# TODO remove this code
# count = 0
#
# def visualize(data_loader, model=None, show_labels=True, overlay=True, interval=0.02, save_dir=None):
#     """
#     Visualize the images in the data_loader.
#
#     TODO: matplotlib using increasing memory when displaying videos
#
#     :param data_loader:
#     :type data_loader: data.dataloader.basedataloader.BaseDataLoader
#     :param model: if show_labels is True and a model is given, the labels predicted by this model are shown.
#     :type model: model.basemodel.BaseModel
#     :param show_labels: show the labels of the images. If no model is given, the ground truth labels are shown.
#     :type show_labels: bool
#     :param overlay: if labels are shown, overlay them on the images or show them in another window.
#     :type overlay: bool
#     :param interval: the interval between showing new frames.
#     :type interval: float
#     :param save_dir: the directory to store the result images.
#     :type save_dir: str
#     """
#     fig, ax1, ax2, ax3, ax4, ax5, ax6 = None, None, None, None, None, None, None
#
#     counter = 0
#
#     for frames, labels, is_start_of_video in iter(data_loader):
#
#         if model is not None and is_start_of_video:
#             model.reset_state()  # RNNs have a state
#
#         # tracker.print_diff()
#         # gc.collect()
#         # print(gc.get_count())
#         # print(gc.garbage[-9:-1])
#         # time.sleep(3)
#
#         if model is not None:
#             output = model(frames)
#             model.repackage_hidden_state()
#             _, predictions = torch.max(output.data, 1)
#         else:
#             predictions = labels
#
#         if labels is not None:
#             print("-> current frame is labeled")
#
#         predictions = predictions if show_labels else None
#         fig, ax1, ax2 = imshow_batch(frames, predictions, class_encoding=data_loader.get_color_encoding(),
#                                      overlay=overlay, fig=fig, axim1=ax1, axim2=ax2, save_dir=save_dir)
#         plt.pause(interval)
#
#         del predictions
#         del frames
#         del labels
#     print("Number of labeled frames: {}".format(counter))
#
#
# def visualize_models(data_loader, models, overlay=True, interval=0.02, save_dir=None):
#     """
#     Visualize the models on the same dataloader.
#
#     TODO: matplotlib using increasing memory when displaying videos
#
#     :param data_loader:
#     :type data_loader: data.dataloader.basedataloader.BaseDataLoader
#     :param models:
#     :type models: List[model.basemodel.BaseModel]
#     :param overlay: if labels are shown, overlay them on the images or show them in another window.
#     :type overlay: bool
#     :param interval: the interval between showing new frames.
#     :type interval: float
#     :param save_dir: the directory to store the result images.
#     :type save_dir: str
#     """
#
#     fig, ax1, ax2, ax3, ax4, ax5, ax6 = None, None, None, None, None, None, None
#
#     counter = 0
#
#     for frames, labels, is_start_of_video in iter(data_loader):
#
#         if is_start_of_video:
#             for model in models:
#                 model.reset_state()  # RNNs have a state
#
#         # tracker.print_diff()
#         # gc.collect()
#         # print(gc.get_count())
#         # print(gc.garbage[-9:-1])
#         # time.sleep(3)
#
#         predictions = []
#         for model in models:
#             output = model(frames)
#             model.repackage_hidden_state()
#             _, prediction = torch.max(output.data, 1)
#             predictions.append(prediction)
#
#         fig, ax1, ax2 = imshow_batch(frames, predictions, class_encoding=data_loader.get_color_encoding(),
#                                      overlay=overlay, fig=fig, axim1=ax1, axim2=ax2, save_dir=save_dir)
#         plt.pause(interval)
#
#         del predictions
#         del frames
#         del labels
#     print("Number of labeled frames: {}".format(counter))



# TODO update this code: integrate in visualizer
# def visualize_flow(data_loader, model=None, show_labels=True, overlay=True, interval=15, save_dir=None):
#     """
#     Visualize the images in the data_loader.
#
#     TODO: reset state of network!
#     TODO: matplotlib using increasing memory when displaying videos
#     TODO: integrate in Visualizer
#
#     :param data_loader:
#     :type data_loader: data.dataloader.basedataloader.BaseDataLoader
#     :param model: if show_labels is True and a model is given, the labels predicted by this model are shown.
#     :type model: model.basemodel.BaseModel
#     :param show_labels: show the labels of the images. If no model is given, the ground truth labels are shown.
#     :type show_labels: bool
#     :param overlay: if labels are shown, overlay them on the images or show them in another window.
#     :type overlay: bool
#     :param interval: the interval between showing new frames.
#     :type interval: float
#     :param save_dir: the directory to store the result images.
#     :type save_dir: str
#     """
#     fig, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = [None, ] * 9
#     fig2, fig3 = None, None
#
#     prev_frames = None
#
#     for frames, labels, _ in iter(data_loader):
#
#         if model is not None:
#             t0 = time.time()
#             output = model(frames)
#             _, predictions = torch.max(output.data, 1)
#             t1 = time.time()
#             print("model inference: {}".format(t1 - t0))
#         else:
#             predictions = labels
#
#         if prev_frames is not None:
#             fig3, ax8, ax9 = imshow_batch(prev_frames, None, class_encoding=data_loader.get_color_encoding(),
#                                overlay=overlay, fig=fig3, axim1=ax8, axim2=None, save_dir=save_dir)
#             fig, ax5, ax6 = imshow_batch(frames, None, class_encoding=data_loader.get_color_encoding(),
#                                overlay=overlay, fig=fig, axim1=ax5, axim2=None, save_dir=save_dir)
#             t2 = time.time()
#             of = opticalflow.optical_flow(prev_frames, frames)
#             t3 = time.time()
#             warped = opticalflow.warp_flow(prev_frames, of)
#             t4 = time.time()
#
#             print("optical flow: {}".format(t3 - t2))
#             print("warping: {}".format(t4 - t3))
#
#             ax3 = imshow_flow(of, ax3)
#             fig2, ax4, ax7 = imshow_batch(warped, None, class_encoding=data_loader.get_color_encoding(),
#                                overlay=overlay, fig=fig2, axim1=ax4, axim2=None, save_dir=save_dir)
#         #
#         # predictions = predictions if show_labels else None
#         # fig, ax1, ax2 = imshow_batch(frames, predictions, class_encoding=data_loader.get_color_encoding(),
#         #                              overlay=overlay, fig=fig, axim1=ax1, axim2=ax2, save_dir=save_dir)
#
#         plt.pause(interval)
#
#         del predictions
#
#         prev_frames = frames
#
# def imshow_flow(flow, ax1=None):
#     """
#     TODO does not support mini batches
#     TODO include in Visualizer
#     """
#     assert flow.shape[0] == 1
#     flow = flow[0]
#     h, w = flow.shape[:2]
#     hsv = np.empty((h, w, 3), dtype=np.uint8)
#     hsv[..., 1] = 255
#
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     hsv[..., 0] = ang * 180 / (np.pi * 2)
#     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#     rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#
#     if ax1 is None:
#         fig, ax1 = plt.subplots(1, 1, figsize=(15, 7))
#     ax1.imshow(rgb/255.)
#
#     return ax1



# TODO remove this code
# def imshow_batch(images, labels=None, class_encoding=None, legend=True, fig=None, axim1=None, axim2=None, overlay=True,
#                  save_dir=None):
#     """
#     TODO deprecated
#     Displays two grids of images. The top grid displays ``images``
#     and the bottom grid ``labels``.
#
#     """
#     images = torchvision.utils.make_grid(images)
#     images_cv2 = transforms.float_tensor_to_cv2_img(images)
#     del images
#
#     if labels is not None:
#         assert class_encoding is not None
#
#         # TODO optimize
#         labels_rgb_tensor = batch_transform(labels.cpu(), [transforms.long_tensor_to_cv2_img,
#                                                            lambda x: _cmap(x, class_encoding),
#                                                            transforms.cv2_img_to_float_tensor])
#         color_predictions = torchvision.utils.make_grid(labels_rgb_tensor)
#         color_predictions = transforms.float_tensor_to_cv2_img(color_predictions)
#
#         if not overlay:
#             if axim1 is None or axim2 is None:
#                 fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
#                 axim1 = ax1.imshow(images_cv2[..., ::-1])
#                 axim2 = ax2.imshow(color_predictions[..., ::-1])
#             else:
#                 axim1.set_data(images_cv2[..., ::-1])
#                 axim2.set_data(color_predictions[..., ::-1])    # may need to be followed by draw()
#         else:
#             if axim1 is None:
#                 fig, ax1 = plt.subplots(1, 1, figsize=(15, 7))
#                 output = (0.6 * images_cv2) + (0.4 * color_predictions)
#                 output = output.astype(np.uint8)
#                 axim1 = ax1.imshow(output[..., ::-1])
#             else:
#                 output = (0.6 * images_cv2) + (0.4 * color_predictions)
#                 output = output.astype(np.uint8)
#                 axim1.set_data(output[..., ::-1])
#
#         if legend:
#             patches = []
#             for class_name, color in class_encoding.items():
#                 patches.append(mpatches.Patch(color=color[::-1], label=class_name))
#             plt.legend(handles=patches, bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
#
#     else:
#         if axim1 is None:
#             fig, ax1 = plt.subplots(1, 1, figsize=(15, 7))
#             axim1 = ax1.imshow(images_cv2[..., ::-1])
#         else:
#             axim1.set_data(images_cv2[..., ::-1])
#
#     if save_dir is not None:
#         global count
#         save_dir = save_dir if save_dir.endswith("/") else save_dir + "/"
#         datautils.ensure_dir(save_dir)
#         fig.savefig(save_dir + "with-state-reset-fig-{}.png".format(count))
#         count += 1
#
#     return fig, axim1, axim2

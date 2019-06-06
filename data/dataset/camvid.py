"""
Based on: https://github.com/davidtvs/PyTorch-ENet

TODO remove duplicate code
"""
from __future__ import annotations
import cv2
from collections import OrderedDict
import logging
import os.path

from utils import datautils as datautils
from utils import transforms as transforms
from data.dataset.datatype import ImageSequence, Video
from data.dataset.dataset import ImgSeqDataset, ImageDataset, VideoDataset
from . import config as config


# Default encoding for pixel value, class name, and class color in BGR 0..1.
COLOR_ENCODING = OrderedDict([('sky', (0.5, 0.5, 0.5)),
                              ('building', (0.5, 0.0, 0.0)),
                              ('pole', (0.75, 0.75, 0.5)),
                              ('road', (0.5, 0.25, 0.5)),
                              ('pavement', (0.23, 0.16, 0.87)),
                              ('tree', (0.5, 0.5, 0.0)),
                              ('sign_symbol', (0.75, 0.5, 0.5)),
                              ('fence', (0.25, 0.25, 0.5)),
                              ('car', (0.25, 0.0, 0.5)),
                              ('pedestrian', (0.25, 0.25, 0.0)),
                              ('bicyclist', (0.0, 0.5, 0.75)),
                              ('unlabeled', (0.0, 0.0, 0.0))])


class CamVid(ImageDataset):
    """
    Dataset containing CamVid images.

    The data is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.
    """

    @staticmethod
    def create(img_size=(360, 480)):
        """
        :param img_size: the images are rescaled to the given size. (height, width)
        :type img_size: (int, int)
        :return: The CamVid training-, validation- and testing sets.
        :rtype: (CamVid, CamVid, CamVid)
        """
        return CamVid(config.CAMVID_TRAIN_DIR, config.CAMVID_TRAIN_LABEL_DIR, img_size, name="train"), \
               CamVid(config.CAMVID_VAL_DIR, config.CAMVID_VAL_LABEL_DIR, img_size, name="val"), \
               CamVid(config.CAMVID_TEST_DIR, config.CAMVID_TEST_LABEL_DIR, img_size, name="test")

    def __init__(self, img_dir, label_dir, img_size=(360, 480), img_file_filter=config.CAMVID_FILE_FILTER,
                 label_file_filter=config.CAMVID_LABEL_FILE_FILTER, name=""):
        """
        :param img_dir:
        :type img_dir: str
        :param label_dir:
        :type label_dir: str
        :param img_size: (height, width)
        :type img_size: (int, int)
        :param img_file_filter:
        :type img_file_filter: (str) -> bool
        :param label_file_filter:
        :type label_file_filter: (str) -> bool
        """
        super(CamVid, self).__init__(name=name)

        self.img_size = img_size

        self.img_dir = img_dir
        self.label_dir = label_dir

        self.img_file_filter = img_file_filter
        self.label_file_filter = label_file_filter

        self.images = datautils.get_files(self.img_dir, name_filter=self.img_file_filter)
        self.labels = datautils.get_files(self.label_dir, name_filter=self.label_file_filter)

    def get_base_directory(self):
        """
        :return: the directory where the dataset is stored.
        :rtype: str
        """
        return config.CAMVID_DIR

    def get_image(self, index):
        """
        Image:
            - type torch.Tensor
            - (channels, height, width)
            - with RGB colors
            - with values in range 0..1
        :param index:
        :type index:
        :return:
        :rtype: torch.Tensor
        """
        img_path = self.images[index]
        img = cv2.imread(img_path)
        h, w = self.img_size    # OpenCV size conventions being annoying
        img = cv2.resize(img, (w, h))
        img = transforms.cv2_img_to_float_tensor(img)
        return img

    def get_label(self, index):
        """
        Label:
            - type torch.Tensor
            - (height, width)
            - with values in range int64
        :param index:
        :type index:
        :return:
        :rtype: torch.Tensor
        """
        label_path = self.labels[index]
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        h, w = self.img_size    # OpenCV size conventions being annoying
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        label = transforms.cv2_img_to_long_tensor(label)

        return label

    def __len__(self):
        """
        :return: the number of images in this dataset.
        :rtype: int
        """
        return len(self.images)

    def get_color_encoding(self):
        """
        :return: The color encoding of the labels of this dataset as an ordered dictionary.
        :rtype: OrderedDict
        """
        return COLOR_ENCODING

    def get_nb_images(self):
        """
        :return: The total number of images in this dataset.
        :rtype: int
        """
        return len(self.images)

    def get_nb_labels(self):
        """
        :return: The total number of labels in this dataset.
        :rtype: int
        """
        return len(self.labels)


class CamVidVideo(VideoDataset):
    """
    Dataset containing CamVid videos.
    """

    @staticmethod
    def create(frame_size=(360, 480)):
        """
        :param frame_size: the frames of the videos are rescaled to the given size.
        :type frame_size: (int, int)
        :return: The CamVid training-, validation- and testing video datasets.
        :rtype: (CamVidVideo, CamVidVideo, CamVidVideo)
        """
        logger = logging.getLogger(CamVidVideo.__name__)

        s0001TP_file = config.CAMVID_DIR + "0001TP.avi"
        s0005VD_file = config.CAMVID_DIR + "0005VD.MXF"
        s0006R0_file = config.CAMVID_DIR + "0006R0.MXF"
        s0016E5_file = config.CAMVID_DIR + "0016E5.MXF"

        # Training sequences
        # ------------------
        logger.info("Creating CamVid training sequences...")
        s0001TP_labels_train = datautils.get_files(config.CAMVID_TRAIN_LABEL_DIR, lambda x: (config.CAMVID_LABEL_FILE_FILTER(x)
                                                                                   and x.startswith("0001TP")))
        s0006R0_labels_train = datautils.get_files(config.CAMVID_TRAIN_LABEL_DIR, lambda x: (config.CAMVID_LABEL_FILE_FILTER(x)
                                                                                   and x.startswith("0006R0")))
        s0016E5_labels_train = datautils.get_files(config.CAMVID_TRAIN_LABEL_DIR, lambda x: (config.CAMVID_LABEL_FILE_FILTER(x)
                                                                                   and x.startswith("0016E5")))

        s0001TP_train = CamVidVideoItem(s0001TP_file, s0001TP_labels_train, video_start_index=2, label_interval=30,
                                        label_start_index=30, frame_size=frame_size)
        s0006R0_train = CamVidVideoItem(s0006R0_file, s0006R0_labels_train, video_start_index=903, label_interval=30,
                                        label_start_index=931, frame_size=frame_size)
        s0016E5_train_1 = CamVidVideoItem(s0016E5_file, s0016E5_labels_train[:68], video_start_index=363,
                                          label_interval=30, label_start_index=391, frame_size=frame_size)
        s0016E5_train_2 = CamVidVideoItem(s0016E5_file, s0016E5_labels_train[68:], video_start_index=4323,
                                          label_interval=30, label_start_index=4351, frame_size=frame_size)

        video_loaders_train = CamVidVideo(s0001TP_train, s0006R0_train, s0016E5_train_1, s0016E5_train_2, name="train")

        # Validation sequences
        # --------------------
        logger.info("Creating CamVid validation sequences...")
        s0016E5_labels_val = datautils.get_files(config.CAMVID_VAL_LABEL_DIR, lambda x: (config.CAMVID_LABEL_FILE_FILTER(x) and
                                                                               x.startswith("0016E5")))

        s0016E5_val = CamVidVideoItem(s0016E5_file, s0016E5_labels_val, video_start_index=7932, label_interval=2,
                                      label_start_index=7960, frame_size=frame_size)

        video_loaders_val = CamVidVideo(s0016E5_val, name="val")

        # Test sequences
        # --------------
        logger.info("Creating CamVid test sequences...")
        s0001TP_labels_test = datautils.get_files(config.CAMVID_TEST_LABEL_DIR, lambda x: (config.CAMVID_LABEL_FILE_FILTER(x) and
                                                                                 x.startswith("0001TP")))
        s0005VD_labels_test = datautils.get_files(config.CAMVID_TEST_LABEL_DIR, lambda x: (config.CAMVID_LABEL_FILE_FILTER(x) and
                                                                                 x.startswith("Seq05VD")))

        s0001TP_test = CamVidVideoItem(s0001TP_file, s0001TP_labels_test, video_start_index=1862, label_interval=30,
                                       label_start_index=1890, frame_size=frame_size)
        s0005VD_test = CamVidVideoItem(s0005VD_file, s0005VD_labels_test[1:], video_start_index=3, label_interval=30,
                                       label_start_index=31, frame_size=frame_size)

        video_loaders_test = CamVidVideo(s0001TP_test, s0005VD_test, name="test")

        return video_loaders_train, video_loaders_val, video_loaders_test

    def __init__(self, *video_loaders: CamVidVideoItem, name=""):
        """
        :param video_loaders: one or more CamVid video loaders.
        :type video_loaders: CamVidVideoItem
        """
        super(CamVidVideo, self).__init__(*video_loaders, name=name)

    def get_base_directory(self):
        """
        :return: the directory where the dataset is stored.
        :rtype: str
        """
        return config.CAMVID_DIR

    def get_color_encoding(self):
        """
        :return: The color encoding of the labels of this dataset as an ordered dictionary in RGB 0..1 format.
        :rtype: OrderedDict
        """
        return COLOR_ENCODING


class CamVidVideoItem(Video):
    """
    Class containing the metadata of a CamVid video file.

    TODO move video loading stuff to a FileVideoLoader class?
    """

    def __init__(self, video_file, label_files, video_start_index=0, video_end_index=-1, label_interval=30,
                 frame_size=(360, 480), label_start_index=0):
        """
        :param video_file:
        :type video_file: str
        :param label_files:
        :type label_files: list[str]
        :param video_start_index:
        :type video_start_index: int
        :param label_interval:
        :type label_interval: int
        :param frame_size: (height, width)
        :type frame_size: (int, int)
        :param label_start_index:
        :type label_start_index: int
        """
        self.cap = None
        self.frame_size = frame_size
        self.video_file = video_file

        name = os.path.split(video_file)[1]
        logger = logging.getLogger(self.__class__.__name__ + "-" + name)

        super(CamVidVideoItem, self).__init__(label_files, label_interval, video_start_index, video_end_index,
                                              label_start_index, logger)

    def _get_total_nb_of_frames(self):
        """
        :return:
        :rtype: int
        """
        self.init_video_file_reader()
        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.close_video_file_reader()
        return int(frame_count)

    def init_video_file_reader(self):
        """
        :return:
        :rtype:
        """
        if self.cap is not None:
            self.close_video_file_reader()

        # Wait for the video file to open
        self.cap = cv2.VideoCapture(self.video_file)
        while not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_file)
            cv2.waitKey(500)
            self.logger.info("waiting for file {} to open...".format(self.video_file))

    def read_next_frame(self, frame_idx):
        """
        Frame:
            - (channels, height, width)
            - with RGB colors
            - with values in range 0..1
        :param frame_idx: the index of the next frame. This function is guaranteed to be called with a frame index that
            is one more than the previous time, or with a frame index 0 after an initialization.
        :type frame_idx: int
        :return:
        :rtype: torch.Tensor
        """
        flag, frame = self.cap.read()
        while not flag:
            # The frame is not ready yet. Try again after waiting 25ms.
            cv2.waitKey(25)
            flag, frame = self.cap.read()

        h, w = self.frame_size  # OpenCV size conventions being annoying
        frame = cv2.resize(frame, (w, h))
        frame = transforms.cv2_img_to_float_tensor(frame)

        return frame

    def read_label_from_file(self, label_path: str):
        """
        Label:
            - (height, width)
            - with values in range int64
        :param label_path: the path of the label to load.
        :type label_path: str
        :return:
        :rtype: torch.Tensor
        """
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        h, w = self.frame_size  # OpenCV size conventions being annoying
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        label = transforms.cv2_img_to_long_tensor(label)
        return label

    def close_video_file_reader(self):
        """
        :return:
        :rtype:
        """
        self.cap.release()
        self.cap = None


class CamVidImgSequence(ImgSeqDataset):

    @staticmethod
    def create(frame_size=(360, 480)):
        """
        :param frame_size: the frames of the videos are rescaled to the given size.
        :type frame_size: (int, int)
        :return: The CamVid training-, validation- and testing video datasets.
        :rtype: (CamVidImgSequence, CamVidImgSequence, CamVidImgSequence)
        """
        logger = logging.getLogger(CamVidImgSequence.__name__)

        s0001TP_dir = config.CAMVID_DIR + "0001TP/"
        s0005VD_dir = config.CAMVID_DIR + "0005VD/"
        s0006R0_dir = config.CAMVID_DIR + "0006R0/"
        s0016E5_dir = config.CAMVID_DIR + "0016E5/"

        s0001TP_file = config.CAMVID_DIR + "0001TP.avi"
        s0005VD_file = config.CAMVID_DIR + "0005VD.MXF"
        s0006R0_file = config.CAMVID_DIR + "0006R0.MXF"
        s0016E5_file = config.CAMVID_DIR + "0016E5.MXF"

        # Training sequences
        # ------------------
        logger.info("Creating CamVid training sequences...")
        s0001TP_labels_train = datautils.get_files(config.CAMVID_TRAIN_LABEL_DIR, lambda x: (config.CAMVID_LABEL_FILE_FILTER(x)
                                                                                   and x.startswith("0001TP")))
        s0006R0_labels_train = datautils.get_files(config.CAMVID_TRAIN_LABEL_DIR, lambda x: (config.CAMVID_LABEL_FILE_FILTER(x)
                                                                                   and x.startswith("0006R0")))
        s0016E5_labels_train = datautils.get_files(config.CAMVID_TRAIN_LABEL_DIR, lambda x: (config.CAMVID_LABEL_FILE_FILTER(x)
                                                                                   and x.startswith("0016E5")))

        # s0001TP_train
        start_index = 2
        label_interval = 30

        img_seq_loaders = []
        for s0001TP_label in s0001TP_labels_train:

            img_files = [s0001TP_dir + "0001TP-{:06d}.png".format(nb) for nb in range(start_index, start_index + label_interval)]
            loader = CamVidImageSequenceItem(img_files, [s0001TP_label, ], video_start_index=0,
                                             label_interval=label_interval, frame_size=frame_size,
                                             label_start_index=28)
            img_seq_loaders.append(loader)
            start_index += label_interval

        # s0006R0_train
        start_index = 903

        for s0006R0_label in s0006R0_labels_train:
            img_files = [s0006R0_dir + "0006R0-{:06d}.png".format(nb) for nb in range(start_index, start_index + label_interval)]
            loader = CamVidImageSequenceItem(img_files, [s0006R0_label, ], video_start_index=0,
                                             label_interval=label_interval, frame_size=frame_size,
                                             label_start_index=28)
            img_seq_loaders.append(loader)
            start_index += label_interval

        # s0016E5_train part 1
        start_index = 363

        for s0016E5_label in s0016E5_labels_train[:68]:
            img_files = [s0016E5_dir + "0016E5-{:06d}.png".format(nb) for nb in range(start_index, start_index + label_interval)]
            loader = CamVidImageSequenceItem(img_files, [s0016E5_label, ], video_start_index=0,
                                             label_interval=label_interval, frame_size=frame_size,
                                             label_start_index=28)
            img_seq_loaders.append(loader)
            start_index += label_interval

        # s0016E5_train part 2
        start_index = 4323

        for s0016E5_label in s0016E5_labels_train[68:]:
            img_files = [s0016E5_dir + "0016E5-{:06d}.png".format(nb) for nb in range(start_index, start_index + label_interval)]
            loader = CamVidImageSequenceItem(img_files, [s0016E5_label, ], video_start_index=0,
                                             label_interval=label_interval, frame_size=frame_size,
                                             label_start_index=28)
            img_seq_loaders.append(loader)
            start_index += label_interval

        video_loaders_train = CamVidImgSequence(*img_seq_loaders, name="train")

        # Validation sequences
        # --------------------
        logger.info("Creating CamVid validation sequences...")
        s0016E5_labels_val = datautils.get_files(config.CAMVID_VAL_LABEL_DIR, lambda x: (config.CAMVID_LABEL_FILE_FILTER(x) and
                                                                               x.startswith("0016E5")))

        s0016E5_val = CamVidVideoItem(s0016E5_file, s0016E5_labels_val, video_start_index=7959, label_interval=2,
                                      label_start_index=7960, frame_size=frame_size)

        video_loaders_val = CamVidVideo(s0016E5_val, name="val")

        # Test sequences
        # --------------
        logger.info("Creating CamVid test sequences...")
        s0001TP_labels_test = datautils.get_files(config.CAMVID_TEST_LABEL_DIR, lambda x: (config.CAMVID_LABEL_FILE_FILTER(x) and
                                                                                 x.startswith("0001TP")))
        s0005VD_labels_test = datautils.get_files(config.CAMVID_TEST_LABEL_DIR, lambda x: (config.CAMVID_LABEL_FILE_FILTER(x) and
                                                                                 x.startswith("Seq05VD")))

        s0001TP_test = CamVidVideoItem(s0001TP_file, s0001TP_labels_test, video_start_index=1862, label_interval=30,
                                       label_start_index=1890, frame_size=frame_size)
        s0005VD_test = CamVidVideoItem(s0005VD_file, s0005VD_labels_test[1:], video_start_index=3, label_interval=30,
                                       label_start_index=31, frame_size=frame_size)

        video_loaders_test = CamVidVideo(s0001TP_test, s0005VD_test, name="test")

        return video_loaders_train, video_loaders_val, video_loaders_test

    def __init__(self, *video_loaders: CamVidImageSequenceItem, name=""):
        """
        :param video_loaders: one or more CamVid video loaders.
        :type video_loaders: CamVidVideoItem
        """
        super(CamVidImgSequence, self).__init__(*video_loaders, name=name)

    def get_base_directory(self):
        """
        :return: the directory where the dataset is stored.
        :rtype: str
        """
        return config.CAMVID_DIR

    def get_color_encoding(self):
        """
        :return: The color encoding of the labels of this dataset as an ordered dictionary in RGB 0..1 format.
        :rtype: OrderedDict
        """
        return COLOR_ENCODING


class CamVidImageSequenceItem(ImageSequence):

    def __init__(self, image_files, label_files, video_start_index=0, video_end_index=-1, label_interval=30,
                 frame_size=(360, 480), label_start_index=0):
        """
        :param image_files:
        :type image_files: list[str]
        :param label_files:
        :type label_files: list[str]
        :param video_start_index:
        :type video_start_index: int
        :param label_interval:
        :type label_interval: int
        :param frame_size: (height, width)
        :type frame_size: (int, int)
        :param label_start_index:
        :type label_start_index: int
        """
        super(CamVidImageSequenceItem, self).__init__(image_files, label_files, label_interval, video_start_index,
                                                      video_end_index, label_start_index)
        self.frame_size = frame_size

    def read_image_from_file(self, image_path: str):
        """
        Frame:
            - (channels, height, width)
            - with RGB colors
            - with values in range 0..1
        :param image_path: the path of the label to load
        :type image_path: str
        :return:
        :rtype: torch.Tensor
        """
        img = cv2.imread(image_path)
        h, w = self.frame_size    # OpenCV size conventions being annoying
        img = cv2.resize(img, (w, h))
        img = transforms.cv2_img_to_float_tensor(img)
        return img

    def read_label_from_file(self, label_path: str) :
        """
        Label:
            - (height, width)
            - with values in range int64
        :param label_path: the path of the label to load.
        :type label_path: str
        :return:
        :rtype: torch.Tensor
        """
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        h, w = self.frame_size    # OpenCV size conventions being annoying
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        label = transforms.cv2_img_to_long_tensor(label)
        return label

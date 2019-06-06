"""
Based on https://github.com/davidtvs/PyTorch-ENet

TODO remove duplicate code
"""
from __future__ import annotations
import cv2
from collections import OrderedDict
import logging
import os.path
from parse import parse

from utils import datautils as datautils
from utils import transforms as transforms
from data.dataset.datatype import ImageSequence
from data.dataset.dataset import ImgSeqDataset, ImageDataset
from . import config as config


# The values associated with the 35 classes
# ['unlabeled','ego vehicle', 'rectification border', 'out of roi', 'static', 'dynamic', 'ground', 'road', 'sidewalk',
# 'parking', 'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup',
# 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan',
# 'trailer', 'train', 'motorcycle', 'bicycle', 'license plate']
FULL_CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, -1)
# The values above are remapped to the following
NEW_CLASSES = (0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 5, 0, 0, 0, 6, 0, 7,
               8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 17, 18, 19, 0)

# Default encoding for pixel value, class name, and class color
COLOR_ENCODING = OrderedDict([
        ('unlabeled', (0., 0., 0.)),
        ('road', (0.5, 0.25, 0.5)),
        ('sidewalk', (1., 0.125, 0.9)),
        ('building', (0.25, 0.25, 0.5)),
        ('wall', (0.4, 0.4, 0.6)),
        ('fence', (0.75, 0.6, 0.6)),
        ('pole', (0.9, 0.5, 0.5)),
        ('traffic_light', (1., 0.65, 0.125)),
        ('traffic_sign', (0.85, 0.85, 0.)),
        ('vegetation', (0.4, 0.65, 0.125)),
        ('terrain', (0.6, 1., 0.6)),
        ('sky', (0.25, 0.5, 0.7)),
        ('person', (0.85, 0.1, 0.25)),
        ('rider', (1., 0., 0.)),
        ('car', (0., 0., 0.55)),
        ('truck', (0., 0., 0.25)),
        ('bus', (0., 0.25, 0.4)),
        ('train', (0., 0.35, 0.4)),
        ('motorcycle', (0., 0., 0.9)),
        ('bicycle', (0.5, 0.05, 0.125))
])


class Cityscapes(ImageDataset):
    """
    Dataset containing Cityscapes images.
    """

    @staticmethod
    def create(img_size=(256, 512)):
        return Cityscapes(config.CITYSCAPES_TRAIN_DIR, config.CITYSCAPES_TRAIN_LABEL_DIR, img_size, name="train"), \
               Cityscapes(config.CITYSCAPES_VAL_DIR, config.CITYSCAPES_VAL_LABEL_DIR, img_size, name="val"), \
               Cityscapes(config.CITYSCAPES_TEST_DIR, config.CITYSCAPES_TEST_LABEL_DIR, img_size, name="test")

    def __init__(self, img_dir, label_dir, img_size=(256, 512), img_file_filter=config.CITYSCAPES_FILE_FILTER,
                 label_file_filter=config.CITYSCAPES_LABEL_FILE_FILTER, name=""):
        super(Cityscapes, self).__init__(name=name)

        self.img_size = img_size

        self.img_dir = img_dir
        self.label_dir = label_dir

        self.img_file_filter = img_file_filter
        self.label_file_filter = label_file_filter

        self.images = datautils.get_files(self.img_dir, name_filter=self.img_file_filter)
        self.labels = datautils.get_files(self.label_dir, name_filter=self.label_file_filter)

        assert len(self.images) == len(self.labels)

    def get_base_directory(self):
        """
        :return: the directory where the dataset is stored.
        :rtype: str
        """
        return config.CITYSCAPES_DIR

    def get_image(self, index):
        """
        Image:
            - (channels, height, width)
            - with RGB colors
            - with values in range 0..1
        :param index:
        :type index: int
        :return:
        :rtype: torch.Tensor
        """
        img_path = self.images[index]
        img = cv2.imread(img_path)
        if 0 in img.shape:
            self.logger.critical("Error while reading image: image {} is empty".format(img_path))
        h, w = self.img_size    # OpenCV size conventions being annoying
        img = cv2.resize(img, (w, h))
        img = transforms.cv2_img_to_float_tensor(img)
        return img

    def get_label(self, index):
        """
        Label:
            - (height, width)
            - with values in range int64
        :param index:
        :type index: int
        :return:
        :rtype: torch.Tensor
        """
        label_path = self.labels[index]
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if 0 in label.shape:
            self.logger.critical("Error while reading label: label {} is empty".format(label_path))
        h, w = self.img_size    # OpenCV size conventions being annoying
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        label = transforms.remap(label, old_values=FULL_CLASSES, new_values=NEW_CLASSES)
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


class CityscapesImgSequence(ImgSeqDataset):

    @staticmethod
    def create(frame_size=(256, 512), short=True):
        """
        :param frame_size: the frames of the videos are rescaled to the given size.
        :type frame_size: (int, int)
        :return: The Cityscapes training-, validation- and testing video datasets.
        :rtype: (CityscapesImgSequence, CityscapesImgSequence, CityscapesImgSequence)
        """
        logger = logging.getLogger(CityscapesImgSequence.__name__)

        # Training sequences
        # ------------------
        if short:
            seq_start_index = 11    # included
            seq_end_index = 21      # not included
        else:
            seq_start_index = 0    # included
            seq_end_index = 30      # not included

        seq_length = 30
        seq_label_pos = 19
        effective_seq_length = seq_end_index - seq_start_index

        logger.info("Creating Cityscapes training sequences...")
        train_seq_loaders = []
        for city in os.listdir(config.CITYSCAPES_TRAIN_LABEL_DIR):
            city_label_dir = os.path.join(config.CITYSCAPES_TRAIN_LABEL_DIR, city)
            assert os.path.isdir(city_label_dir)
            for label_file in datautils.get_files(city_label_dir, config.CITYSCAPES_LABEL_FILE_FILTER):
                parsed = parse("{}_{}_{:d}_{}.png", label_file)
                _, seq_nb, frame_nb, _ = parsed
                frame_folder = os.path.join(config.CITYSCAPES_TRAIN_SEQ_DIR, city) + "/"
                frame_files = [frame_folder + "{}_{}_{:06d}_leftImg8bit.png".format(city, seq_nb, i)
                               for i in range(frame_nb - seq_label_pos,
                                              frame_nb - seq_label_pos + seq_length)]

                loader = CityscapesImageSequenceItem(frame_files, [label_file], label_interval=effective_seq_length,
                                                     video_start_index=seq_start_index, video_end_index=seq_end_index,
                                                     frame_size=frame_size, label_start_index=seq_label_pos)
                train_seq_loaders.append(loader)

        video_loaders_train = CityscapesImgSequence(*train_seq_loaders, name="train")

        # Validation sequences
        # ------------------
        logger.info("Creating Cityscapes validation sequences...")
        val_seq_loaders = []
        for city in os.listdir(config.CITYSCAPES_VAL_LABEL_DIR):
            city_label_dir = os.path.join(config.CITYSCAPES_VAL_LABEL_DIR, city)
            assert os.path.isdir(city_label_dir)
            for label_file in datautils.get_files(city_label_dir, config.CITYSCAPES_LABEL_FILE_FILTER):
                parsed = parse("{}_{}_{:d}_{}.png", label_file)
                _, seq_nb, frame_nb, _ = parsed
                frame_folder = os.path.join(config.CITYSCAPES_VAL_SEQ_DIR, city) + "/"
                frame_files = [frame_folder + "{}_{}_{:06d}_leftImg8bit.png".format(city, seq_nb, i)
                               for i in range(frame_nb - seq_label_pos,
                                              frame_nb - seq_label_pos + seq_length)]

                loader = CityscapesImageSequenceItem(frame_files, [label_file], label_interval=effective_seq_length,
                                                     video_start_index=seq_start_index, video_end_index=seq_end_index,
                                                     frame_size=frame_size, label_start_index=seq_label_pos)
                val_seq_loaders.append(loader)

        video_loaders_val = CityscapesImgSequence(*val_seq_loaders, name="train")

        # Test sequences
        # ------------------
        logger.info("Creating Cityscapes test sequences...")
        test_seq_loaders = []
        for city in os.listdir(config.CITYSCAPES_TEST_LABEL_DIR):
            city_label_dir = os.path.join(config.CITYSCAPES_TEST_LABEL_DIR, city)
            assert os.path.isdir(city_label_dir)
            for label_file in datautils.get_files(city_label_dir, config.CITYSCAPES_LABEL_FILE_FILTER):
                parsed = parse("{}_{}_{:d}_{}.png", label_file)
                _, seq_nb, frame_nb, _ = parsed
                frame_folder = os.path.join(config.CITYSCAPES_TEST_SEQ_DIR, city) + "/"
                frame_files = [frame_folder + "{}_{}_{:06d}_leftImg8bit.png".format(city, seq_nb, i)
                               for i in range(frame_nb - seq_label_pos,
                                              frame_nb - seq_label_pos + seq_length)]

                loader = CityscapesImageSequenceItem(frame_files, [label_file], label_interval=effective_seq_length,
                                                     video_start_index=seq_start_index, video_end_index=seq_end_index,
                                                     frame_size=frame_size, label_start_index=seq_label_pos)
                test_seq_loaders.append(loader)

        video_loaders_test = CityscapesImgSequence(*test_seq_loaders, name="test")

        return video_loaders_train, video_loaders_val, video_loaders_test

    def __init__(self, *image_sequences: CityscapesImageSequenceItem, name=""):
        """
        :param image_sequences: one or more CamVid image sequences.
        :type image_sequences: CityscapesImageSequenceItem
        """
        super(CityscapesImgSequence, self).__init__(*image_sequences, name=name)

    def get_base_directory(self):
        """
        :return: the directory where the dataset is stored.
        :rtype: str
        """
        return config.CITYSCAPES_DIR

    def get_color_encoding(self):
        """
        :return: The color encoding of the labels of this dataset as an ordered dictionary in RGB 0..1 format.
        :rtype: OrderedDict
        """
        return COLOR_ENCODING


class CityscapesImageSequenceItem(ImageSequence):

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
        super(CityscapesImageSequenceItem, self).__init__(image_files, label_files, label_interval, video_start_index,
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
        h, w = self.frame_size  # OpenCV size conventions being annoying
        img = cv2.resize(img, (w, h))
        img = transforms.cv2_img_to_float_tensor(img)
        return img

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
        h, w = self.frame_size    # OpenCV size conventions being annoying
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        label = transforms.remap(label, old_values=FULL_CLASSES, new_values=NEW_CLASSES)
        label = transforms.cv2_img_to_long_tensor(label)
        return label

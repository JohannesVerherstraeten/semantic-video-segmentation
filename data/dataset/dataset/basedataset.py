"""
Internally, all images should be in the following format:
    - (channels, height, width)
    - with RGB colors
    - with values in range 0..1
"""
from __future__ import annotations
from typing import *
import torch.utils.data
import logging
from collections import OrderedDict


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, name: str = ""):
        """
        :param name: name of this dataset. Usually "train", "val", or "test". Only for logging/debugging purpose.
        :type name: str
        """
        super(BaseDataset, self).__init__()
        self.name = name
        self.logger = logging.getLogger(self.__class__.__name__ + ("-{}".format(name) if len(name) > 0 else ""))

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Returns elements of the form: (image, label, is_start_of_sequence)
        Image:
            - (channels, height, width)
            - with RGB colors
            - with values in range 0..1
        Label:
            - (height, width)
            - with values in range int64
        :param index: index of the item in the dataset to retrieve.
        :type index: int
        :return: A pair of tensors: image and ground-truth.
        :rtype: (torch.Tensor, torch.Tensor)
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def get_base_directory(self) -> str:
        """
        :return: the directory where the dataset is stored.
        :rtype: str
        """
        raise NotImplementedError

    def get_color_encoding(self) -> OrderedDict:
        """
        :return: the color encoding of the labels of this dataset.
        :rtype: OrderedDict
        """
        raise NotImplementedError

    def get_nb_classes(self) -> int:
        """
        :return: the number of classes in this dataset.
        :rtype: int
        """
        return len(self.get_color_encoding())

    def get_nb_images(self) -> int:
        """
        :return: The total number of images in this dataset.
        :rtype: int
        """
        raise NotImplementedError

    def get_nb_labels(self) -> int:
        """
        :return: The total number of labels in this dataset.
        :rtype: int
        """
        raise NotImplementedError

    def get_labels(self) -> Iterator[torch.Tensor]:
        """
        :return: an iterator over the labels in this dataset.
        :rtype: Iterator[torch.Tensor]
        """
        raise NotImplementedError

    def create_dataloader(self, **kwargs):
        """
        :return: a data loader instance that can load this dataset.
        :rtype: data.dataloader.basedataloader.BaseDataLoader
        """
        raise NotImplementedError

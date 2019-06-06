from __future__ import annotations
from typing import *
import torch
from collections import OrderedDict

import data.dataloader.imagedataloader as imagedataloader
from data.dataset.dataset.basedataset import BaseDataset


class ImageDataset(BaseDataset):
    """
    Base class for an image dataset.
    """

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
        return self.get_image(index), self.get_label(index), True

    def get_image(self, index) -> torch.Tensor:
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
        raise NotImplementedError

    def get_label(self, index) -> torch.Tensor:
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
        raise NotImplementedError

    def __len__(self) -> int:
        """
        :return: number of images in the dataset.
        :rtype: int
        """
        return self.get_nb_images()

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
        class LabelIterator(object):

            def __init__(self, image_dataset):
                """
                :param image_dataset:
                :type image_dataset: ImageDataset
                """
                self.image_dataset = image_dataset
                self.index = 0

            def __next__(self):
                """
                :return:
                :rtype: torch.Tensor
                """
                if self.index == len(self.image_dataset):
                    raise StopIteration
                else:
                    result = self.image_dataset.get_label(self.index)
                    self.index += 1
                    return result

            def __len__(self):
                return len(self.image_dataset)

            def __iter__(self):
                return self

        return LabelIterator(self)

    def create_dataloader(self, **kwargs) -> imagedataloader.ImageDataLoader:
        """
        :return: a data loader instance that can load this dataset.
        :rtype: data.dataloader.imagedataloader.ImageDataLoader
        """
        return imagedataloader.ImageDataLoader(self, **kwargs)

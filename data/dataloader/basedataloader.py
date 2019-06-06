from __future__ import annotations
import torch
from typing import *
from collections import OrderedDict
from torch.utils.data import dataloader as dl


def collate_to_batch(batch: List[Tuple[torch.Tensor, Optional[torch.Tensor], bool]]) \
        -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
    """
    Method that merges items from the dataset to a batch.

    Images from an image dataset are not part of a sequence, (otherwise, make a video dataset).
    For that reason, the network's internal state must always be reset before processing the image.
    Therefore, the is_start_of_sequence flag is always True.

    :param batch:
    :type batch: list[(torch.Tensor, torch.Tensor)]
    :return: (image_batch, label_batch, is_start_of_sequence)
    :rtype: (torch.Tensor, Torch.Tensor, bool)
    """
    images_tuple, labels_tuple, bool_tuple = zip(*batch)    # zip(*list) is opposite of zip(list)
    assert all(bool_tuple) or not any(bool_tuple)           # all frames must be start of sequence, or none of them
    none_labels_tuple = [label is None for label in labels_tuple]
    assert all(none_labels_tuple) or not any(none_labels_tuple)     # all labels must be None, or none of them

    images_batch = dl.default_collate(images_tuple)
    if not any(none_labels_tuple):
        labels_batch = dl.default_collate(labels_tuple)
    else:
        labels_batch = None
    return images_batch, labels_batch, all(bool_tuple)


class BaseDataLoader(object):
    """
    Data loader for datasets.

    Combines a dataset and a sampler.

    Iterating over this data loader returns elements of the form:
    (images, labels, is_start_of_sequence)
    with type
    (torch.Tensor, Torch.Tensor, bool)

    Image:
        - (batch_size, channels, height, width)
        - with RGB colors
        - with values in range 0..1
    Label:
        - (batch_size, height, width)
        - with values in range int64

    The is_start_of_sequence flag indicates whether the state of the network should be reset
    before processing this image. In case the dataset only consists of images that should not be
    seen as a sequence, this flag must always be True. In case the dataset consists of sequences of
    images, the flag must be True for the first image of the sequences.
    """

    def __init__(self):
        super(BaseDataLoader, self).__init__()

    def get_dataset(self):
        """
        Get the dataset of this dataloader.

        :return:
        :rtype: data.dataset.dataset.basedataset.BaseDataset
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Optional[torch.Tensor], bool]]:
        """
        An iterator that returns elements of the following form:
        (images, labels, is_start_of_sequence)
        with type:
        (torch.Tensor, Optional[torch.Tensor], bool)

        Image:
            - (batch_size, channels, height, width)
            - with RGB colors
            - with values in range 0..1
        Label:
            - (batch_size, height, width)
            - with values in range int64

        The is_start_of_sequence flag indicates whether the state of the network should be reset
        before processing this image. In case the dataset only consists of images that should not be
        seen as a sequence, this flag must always be True. In case the dataset consists of sequences of
        images, the flag must be True for the first image of the sequences.

        :return: an iterator over this dataloader.
        :rtype: Iterator[(torch.Tensor, Optional[torch.Tensor], bool)]
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Number of elements returned by the iterator.

        As the iterator may return batches of images and labels, this is not necessarily equal to get_nb_images().
        """
        raise NotImplementedError

    def get_nb_images(self) -> int:
        """
        Number of images in the dataset.
        """
        return self.get_dataset().get_nb_images()

    def get_nb_labels(self) -> int:
        """
        Number of labels in the dataset.
        """
        return self.get_dataset().get_nb_labels()

    def get_color_encoding(self) -> OrderedDict:
        """
        :return: the color encoding of the labels in this dataloader.
        :rtype: OrderedDict
        """
        return self.get_dataset().get_color_encoding()

    def get_nb_classes(self) -> int:
        """
        :return: the number of classes in this dataloader.
        :rtype: int
        """
        return len(self.get_color_encoding())

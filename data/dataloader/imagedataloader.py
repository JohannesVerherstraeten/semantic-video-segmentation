from __future__ import annotations
import torch
from torch.utils.data import dataloader as dl
from typing import *

from .basedataloader import BaseDataLoader, collate_to_batch
import data.dataset.dataset.imagedataset as imagedataset


class ImageDataLoader(BaseDataLoader):
    """
    Data loader for image datasets. Basically wraps the Pytorch DataLoader class.

    Combines a dataset and a sampler.

    Iterating over this dataloader returns elements of the form:
    (image_batch, label_batch, is_start_of_sequence)
    with type
    (torch.Tensor, Torch.Tensor, True)

    Image:
        - (batch_size, channels, height, width)
        - with RGB colors
        - with values in range 0..1
    Label:
        - (batch_size, height, width)
        - with values in range int64

    The is_start_of_sequence flag indicates whether the state of the network should be reset
    before processing this image. In this case the dataset only consists of images that should not be
    seen as a sequence, so this flag is always True. The network won't keep a state when processing
    these images.
    """

    def __init__(self, dataset: imagedataset.ImageDataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, drop_last=False):
        """
        :param dataset:
        :type dataset: ImageDataset
        :param batch_size:
        :type batch_size: int
        :param shuffle:
        :type shuffle: bool
        :param sampler:
        :type sampler: Optional[torch.utils.data.sampler.Sampler]
        :param batch_sampler:
        :type batch_sampler: Optional[torch.utils.data.sampler.Sampler]
        :param num_workers:
        :type num_workers: int
        :param drop_last:
        :type drop_last: bool
        """
        super(ImageDataLoader, self).__init__()
        self.dataset = dataset
        self.data_loader = dl.DataLoader(dataset, batch_size, shuffle, sampler, batch_sampler,
                                         num_workers, collate_fn=collate_to_batch, drop_last=drop_last)

    def get_dataset(self) -> imagedataset.ImageDataset:
        """
        Get the dataset of this dataloader.
        """
        return self.dataset

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
        return iter(self.data_loader)

    def __len__(self) -> int:
        """
        Number of elements returned by the iterator.

        As the iterator may return batches of images and labels, this is not necessarily equal to get_nb_images().
        """
        return len(self.data_loader)

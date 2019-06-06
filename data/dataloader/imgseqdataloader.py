from __future__ import annotations
import torch
from torch.utils.data import dataloader as dl
from typing import *

from .basedataloader import BaseDataLoader, collate_to_batch
import data.dataset.dataset.imgseqdataset as imgseqdataset
import data.dataloader.batchsampler.imgseqbatchsampler as imgseqbatchsampler


class ImgSeqDataLoader(BaseDataLoader):
    """
    Data loader for image sequence datasets.

    Combines a dataset and a sampler.

    Iterating over this dataloader returns elements of the form:
    (image_batch, label_batch, is_start_of_sequence)
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
    before processing this image. This is the case when the batch contains images that are the
    first of their sequence.
    """

    def __init__(self, dataset: imgseqdataset.ImgSeqDataset, batch_size=1, shuffle=False, num_workers=0,
                 drop_last=False, use_workers=False):
        """
        :param dataset:
        :type dataset: ImgSeqDataset
        :param batch_size:
        :type batch_size: int
        :param shuffle: whether the image sequences should appear in order or not.
        :type shuffle: bool
        :param num_workers: can be overridden by use_workers
        :type num_workers: int
        :param drop_last:
        :type drop_last: bool
        """
        super(ImgSeqDataLoader, self).__init__()
        self.dataset = dataset

        if drop_last:
            raise NotImplementedError("Drop last for image sequence batches not implemented yet")

        batch_sampler = imgseqbatchsampler.ImgSeqBatchSampler(dataset, batch_size, shuffle)
        if use_workers:
            num_workers = (batch_size + 1) // 2

        self.data_loader = dl.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers,
                                         collate_fn=collate_to_batch)

    def get_dataset(self) -> imgseqdataset.ImgSeqDataset:
        """
        Get the dataset of this dataloader.

        :return:
        :rtype: data.dataset.dataset.imgseqdataset.ImgSeqDataset
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

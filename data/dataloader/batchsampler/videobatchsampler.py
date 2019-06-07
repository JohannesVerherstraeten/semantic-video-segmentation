from __future__ import annotations
from typing import *

import data.dataset.dataset.videodataset as videodataset
from .basevideobatchsampler import BaseVideoBatchSampler, BaseVideoBatchSamplerIter


class VideoBatchSampler(BaseVideoBatchSampler):
    """
    Creates batches of videos that have the same label interval.

    Videos in batches are ordered by decreasing length.
    """
    def __init__(self, dataset: videodataset.VideoDataset, batch_size: int, shuffle: bool):
        """
        :param dataset:
        :type dataset: VideoDataset
        :param batch_size:
        :type batch_size: int
        :param shuffle:
        :type shuffle: bool
        """
        super(VideoBatchSampler, self).__init__(dataset, batch_size, shuffle)

    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        return VideoBatchSamplerIter(self)

    def __len__(self) -> int:
        raise NotImplementedError


class VideoBatchSamplerIter(BaseVideoBatchSamplerIter):

    def __init__(self, videobatchsampler: VideoBatchSampler):
        super(VideoBatchSamplerIter, self).__init__(videobatchsampler)
        self.result_iterator = iter(self.batches)

    def __next__(self) -> Tuple[int, ...]:
        return next(self.result_iterator)

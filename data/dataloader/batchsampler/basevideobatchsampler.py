"""
TODO avoid duplicate code with imgseqbatchsampler
"""
from __future__ import annotations
import torch.utils.data
from typing import *
import logging
import numpy.random

from data.dataset.dataset.basevideodataset import BaseVideoDataset
from data.dataset.datatype.basevideo import BaseVideo


class BaseVideoBatchSampler(torch.utils.data.Sampler):
    """
    Base class for video batch samplers.

    Creates batches of video frames, of which the videos have the same label interval.

    Videos in in batches are ordered by decreasing length.

    TODO check position of label in sequence?
    """
    def __init__(self, videodataset: BaseVideoDataset, batch_size: int, shuffle: bool):
        """
        :param videodataset:
        :type videodataset: BaseVideoDataset
        :param batch_size:
        :type batch_size: int
        :param shuffle:
        :type shuffle: bool
        """
        super(BaseVideoBatchSampler, self).__init__(videodataset)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset = videodataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.idx_to_length = [len(video) for video in self.dataset.get_videos()]

    def group_videos_as_batch(self):
        # maps an integer to the videos having this integer as label interval
        intervals_to_idx = self.__sort_by_label_interval(self.dataset.get_videos())

        # determines which videos go together in a batch.
        batches = self.__create_batches_per_interval(intervals_to_idx)
        return batches

    def __sort_by_label_interval(self, videos: Tuple[BaseVideo, ...]) -> Dict[int, Tuple[int, ...]]:
        intervals_to_idx = dict()
        for idx, video in enumerate(videos):
            videos = intervals_to_idx.setdefault(video.get_label_interval(), [])
            videos.append(idx)

        if len(intervals_to_idx) > 1:
            self.logger.debug("Video dataset contains videos with different label intervals.")

        return intervals_to_idx

    def __create_batches_per_interval(self, intervals_to_idx: Dict[int, Tuple[int, ...]]) -> List[Tuple[int, ...]]:
        batches = []
        for label_interval, video_idxs in intervals_to_idx.items():

            if len(video_idxs) % self.batch_size != 0:
                self.logger.warning("Number of video_loaders ({}) with label interval {} doesn't fit into batch "
                                    "size {}.".format(len(video_idxs), label_interval, self.batch_size))
            video_idxs = list(video_idxs)
            if self.shuffle:
                video_idxs = numpy.random.permutation(video_idxs)
            else:
                # Sort videos by length in decreasing order
                video_idxs.sort(key=lambda x: self.idx_to_length[x], reverse=True)

            batch = []
            for i, video_idx in enumerate(video_idxs):
                batch.append(video_idx)
                if len(batch) == self.batch_size or i == len(video_idxs) - 1:
                    # sort videos in batch by length in decreasing order
                    batch.sort(key=lambda x: self.idx_to_length[x], reverse=True)
                    batches.append(tuple(batch))
                    batch = []

        return batches

    def __iter__(self) -> BaseVideoBatchSamplerIter:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class BaseVideoBatchSamplerIter(Iterator):

    def __init__(self, basevideobatchsampler: BaseVideoBatchSampler):
        self.batchsampler = basevideobatchsampler
        self.batches = basevideobatchsampler.group_videos_as_batch()

    def __next__(self) -> Tuple:
        raise NotImplementedError

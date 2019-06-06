from __future__ import annotations
from typing import *

import data.dataset.dataset.imgseqdataset as imgseqdataset
import data.dataset.datatype.basevideo as basevideo
from .basevideobatchsampler import BaseVideoBatchSampler, BaseVideoBatchSamplerIter


class ImgSeqBatchSampler(BaseVideoBatchSampler):
    """
    Creates batches of video frames, of which the videos have the same label interval.

    Videos in batches are ordered by decreasing length.

    An iterator over this batch sampler returns indices of the following format.
    Let N be the batch size, and M = max(video lengths of video 1 to N)

    Item 1: ((video_1, frame_1_1), (video_2, frame_2_1), ... (video_N, frame_N_1))
    Item 2: ((video_1, frame_1_2), (video_2, frame_2_2), ... (video_N, frame_N_2))
    ...
    Item M+1: ((video_N+1, frame_N+1_1), (video_N+2, frame_N+2_1), ... (video_2N, frame_2N_1))
    Item M+2: ((video_N+1, frame_N+1_2), (video_N+2, frame_N+2_2), ... (video_2N, frame_2N_2))
    ...

    Note that the returned values are indices, and not the videos and frames itself.
    """

    def __init__(self, dataset: imgseqdataset.ImgSeqDataset, batch_size: int, shuffle: bool):
        """
        :param dataset:
        :type dataset: ImgSeqDataset
        :param batch_size:
        :type batch_size: int
        :param shuffle:
        :type shuffle: bool
        """
        super(ImgSeqBatchSampler, self).__init__(dataset, batch_size, shuffle)

    def create_result_index_pairs(self, batches: List[Tuple[int, ...]]) -> List[Tuple[Tuple[int, int], ...]]:
        """
        Determines how many frames should be loader for each batch
        Determines the final pairs (video_index, frame_index) that will be returned by this batch sampler
        """
        result = []
        for batch in batches:
            max_nb_frames = max(self.idx_to_length[video_idx] for video_idx in batch)

            for frame_idx in range(max_nb_frames):
                batch_with_frame_indices = []

                for video_idx in batch:
                    nb_frames = self.idx_to_length[video_idx]
                    video: basevideo.BaseVideo = self.dataset.get_videos()[video_idx]
                    start_idx = video.video_start_index
                    if nb_frames > frame_idx:
                        batch_with_frame_indices.append((video_idx, frame_idx + start_idx))
                result.append(tuple(batch_with_frame_indices))

        return result

    def __iter__(self) -> Iterator[Tuple[Tuple[int, int], ...]]:
        return ImgSeqBatchSamplerIter(self)

    def __len__(self) -> int:
        raise NotImplementedError


class ImgSeqBatchSamplerIter(BaseVideoBatchSamplerIter):

    def __init__(self, imgseqbatchsampler: ImgSeqBatchSampler):
        super(ImgSeqBatchSamplerIter, self).__init__(imgseqbatchsampler)
        result = imgseqbatchsampler.create_result_index_pairs(self.batches)
        self.result_iterator = iter(result)

    def __next__(self) -> Tuple[Tuple[int, int], ...]:
        result = next(self.result_iterator)
        return result

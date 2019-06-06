from __future__ import annotations
from typing import *

from .threadedvideoiterator import ThreadedVideoIterator
from .videoiterator import VideoIterator
from data.dataset.datatype.video import Video
from ..basedataloader import collate_to_batch


def have_equal_label_interval(video_list: List[Video]) -> bool:
    if len(video_list) < 2:
        return True
    else:
        interval_list = [video.get_label_interval() for video in video_list]
        first = interval_list[0]
        return all(first == rest for rest in interval_list)


def are_sorted_by_decreasing_length(video_list: List[Video]) -> bool:
    if len(video_list) < 2:
        return True
    else:
        lenght_list = [len(video) for video in video_list]
        return all(lenght_list[i] >= lenght_list[i + 1] for i in range(len(lenght_list) - 1))


class BatchVideoIterator(object):
    """
    A video iterator for fetching frames and their corresponding labels from the videos in the mini-batch.
    """
    def __init__(self, *videos: Video, use_workers: bool = False):
        """
        :param videos:
        :type videos: Video
        """
        videos = list(videos)
        if not have_equal_label_interval(videos):
            raise RuntimeError("All videos in a batch must have the same label interval")
        if not are_sorted_by_decreasing_length(videos):
            raise RuntimeError("Videos in a batch must be sorted by decreasing length")

        iterator_type = ThreadedVideoIterator if use_workers else VideoIterator

        self.video_iters = [iterator_type(video) for video in videos]
        self.are_open = len(self.video_iters)

    def __next__(self):
        """
        Returns the next frame of the video loader with the corresponding label if it exists.
        Result: (frame_tensor, Optional[label_tensor])
        frame_tensor:
            - (batch_size, channels, height, width)
            - with RGB colors
            - with values in range 0..1
        label_tensor:
            - (batch_size, height, width)
            - with values in range int64
        :return:
        :rtype: (torch.Tensor, torch.Tensor)
        """
        if self.are_open == 0:
            raise StopIteration

        batch_results = []
        for i in range(self.are_open):
            video_iter = self.video_iters[i]
            try:
                result_i = next(video_iter)
            except StopIteration:
                self.are_open = i
                if self.are_open == 0:
                    raise StopIteration
                else:
                    break
            else:
                batch_results.append(result_i)

        return collate_to_batch(batch_results)

    def close(self):
        """
        Close and release the resources of this video iterator.
        """
        [videoiter.close() for videoiter in self.video_iters]

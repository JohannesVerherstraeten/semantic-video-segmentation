from __future__ import annotations
from typing import *
from collections import OrderedDict

import data.dataloader.videodataloader as videodataloader
from .basevideodataset import BaseVideoDataset
from data.dataset.datatype.video import Video


class VideoDataset(BaseVideoDataset):
    """
    Base class for a video dataset.

    A video dataset contains of a video loader for each video.
    """

    def __init__(self, *videos: Video, name: str = ""):
        """
        :param videos:
        :type videos: Video
        """
        super(VideoDataset, self).__init__(name=name)
        self.videos = videos

    def get_videos(self) -> Tuple[Video, ...]:
        """
        Returns all BaseVideo items in this dataset.
        """
        return self.videos

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

    def create_dataloader(self, **kwargs) -> videodataloader.VideoDataLoader:
        """
        :return: a data loader instance that can load this dataset.
        :rtype: data.dataloader.videodataloader.VideoDataLoader
        """
        return videodataloader.VideoDataLoader(self, **kwargs)

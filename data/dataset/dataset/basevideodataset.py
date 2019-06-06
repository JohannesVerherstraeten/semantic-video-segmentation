from __future__ import annotations
from .basedataset import BaseDataset
import itertools
from typing import *
import torch
from collections import OrderedDict

from data.dataset.datatype.basevideo import BaseVideo
from data.dataloader.basedataloader import BaseDataLoader


class BaseVideoDataset(BaseDataset):
    """
    Base class for image sequence dataset.
    """
    def __init__(self, name=""):
        super(BaseVideoDataset, self).__init__(name=name)

    def get_videos(self) -> Tuple[BaseVideo, ...]:
        """
        Returns all BaseVideo items in this dataset.
        """
        raise NotImplementedError

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Accepts double indices: (video_index, frame_index)

        Returns elements of the form: (image, label, is_start_of_sequence)
        Image:
            - (channels, height, width)
            - with RGB colors
            - with values in range 0..1
        Label:
            - (height, width)
            - with values in range int64
        :param index: index of the video frame in the dataset to retrieve: (video_index, frame_index).
        :type index: Tuple[int, int]
        :return: A pair of tensors: image and ground-truth, and a is_start_of_sequence_flag.
        :rtype: (torch.Tensor, torch.Tensor, bool)
        """
        assert len(index) == 2
        video_index, frame_index = index
        return self.get_videos()[video_index].read_next(frame_index)

    def get_image(self, index) -> torch.Tensor:
        """
        Accepts double indices: (video_index, frame_index)

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
        assert len(index) == 2
        video_index, frame_index = index
        video: BaseVideo = self.get_videos()[video_index]
        return video.read_next_frame(frame_index)

    def get_label(self, index) -> torch.Tensor:
        """
        Accepts double indices: (video_index, frame_index)

        Label:
            - type torch.Tensor
            - (height, width)
            - with values in range int64
        :param index:
        :type index:
        :return:
        :rtype: torch.Tensor
        """
        assert len(index) == 2
        video_index, frame_index = index
        video: BaseVideo = self.get_videos()[video_index]
        return video.read_next_label(frame_index)

    def __len__(self) -> int:
        """
        :return: number of image sequences in the dataset.
        :rtype: int
        """
        return self.get_nb_videos()

    def get_base_directory(self) -> str:
        """
        :return: the directory where the dataset is stored.
        :rtype: str
        """
        raise NotImplementedError

    def get_nb_videos(self) -> int:
        """
        :return: number of image sequences in this dataset.
        :rtype:
        """
        return len(self.get_videos())

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
        total = 0
        for video in self.get_videos():
            total += len(video)
        return total

    def get_nb_labels(self) -> int:
        """
        :return: The total number of labels in this dataset.
        :rtype: int
        """
        total = 0
        for video in self.get_videos():
            total += video.get_nb_labels()
        return total

    def get_labels(self) -> Iterator[torch.Tensor]:
        """
        :return: an iterator over the labels in this dataset.
        :rtype: Iterator[torch.Tensor]
        """
        return itertools.chain(*[video.labels() for video in self.get_videos()])

    def create_dataloader(self, **kwargs) -> BaseDataLoader:
        """
        :return: a data loader instance that can load this dataset.
        :rtype:
        """
        raise NotImplementedError

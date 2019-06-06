from __future__ import annotations
from .basevideodataset import BaseVideoDataset
from data.dataset.datatype.imagesequence import ImageSequence
import data.dataloader.imgseqdataloader as imgseqdataloader
from typing import *

from collections import OrderedDict


class ImgSeqDataset(BaseVideoDataset):
    """
    Base class for image sequence dataset.
    """
    def __init__(self, *image_sequences: ImageSequence, name: str = ""):
        """
        :param video_files:
        :type video_files: ImageSequence
        """
        super(ImgSeqDataset, self).__init__(name=name)
        self.image_sequences = image_sequences

    def get_videos(self) -> Tuple[ImageSequence, ...]:
        """
        Returns all BaseVideo items in this dataset.
        """
        return self.image_sequences

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

    def create_dataloader(self, **kwargs) -> imgseqdataloader.ImgSeqDataLoader:
        """
        :return: a data loader instance that can load this dataset.
        :rtype: data.dataloader.imgseqdataloader.ImgSeqDataLoader
        """
        return imgseqdataloader.ImgSeqDataLoader(self, **kwargs)

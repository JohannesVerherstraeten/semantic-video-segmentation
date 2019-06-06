from .basevideo import BaseVideo
from typing import *
import logging
import torch


class ImageSequence(BaseVideo):

    def __init__(self, image_files: List[str], label_files: List[str], label_interval: int, video_start_index: int = 0,
                 video_end_index: int = -1,  label_start_index: int = 0, logger: logging.Logger = None):
        self.image_files = image_files
        super(ImageSequence, self).__init__(label_files, label_interval, video_start_index, video_end_index,
                                            label_start_index, logger)

    def _get_total_nb_of_frames(self) -> int:
        """
        :return: the total number of frames, including the ones that will be skipped.
        :rtype: int
        """
        return len(self.image_files)

    def init_video_file_reader(self):
        """
        :return:
        :rtype:
        """
        pass

    def read_next_frame(self, frame_idx: int) -> torch.Tensor:
        """
        Frame:
            - (channels, height, width)
            - with RGB colors
            - with values in range 0..1
        :param frame_idx: the index of the next frame. This function is guaranteed to be called with a frame index that
            is one more than the previous time, or with a frame index 0 after an initialize.
        :type frame_idx: int
        :return:
        :rtype: torch.Tensor
        """
        return self.read_image_from_file(self.image_files[frame_idx])

    def read_image_from_file(self, image_path: str) -> Optional[torch.Tensor]:
        """
        Frame:
            - (channels, height, width)
            - with RGB colors
            - with values in range 0..1
        :param image_path: the path of the label to load
        :type image_path: str
        :return:
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def read_label_from_file(self, label_path: str) -> Optional[torch.Tensor]:
        """
        Label:
            - (height, width)
            - with values in range int64
        :param label_path: the path of the label to load.
        :type label_path: str
        :return:
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def close_video_file_reader(self):
        """
        :return:
        :rtype:
        """
        pass

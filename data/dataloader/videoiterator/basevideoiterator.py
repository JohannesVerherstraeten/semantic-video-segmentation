from __future__ import annotations
import logging
from typing import *
import torch

from data.dataset.datatype.video import Video


class BaseVideoIterator(object):
    """
    Base class for video iterators.

    A video iterator fetches frames and the corresponding label(s) from a video.
    """

    def __init__(self, video: Video):
        """
        :param video:
        :type video: Video
        """
        super(BaseVideoIterator, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.video = video
        self.label_files = video.label_files
        self.video_start_index = video.video_start_index
        self.label_start_index = video.label_start_index
        self.label_interval = video.label_interval

        self.frame_count = video.frame_count
        self.label_count = video.label_count

        self.current_video_index = 0

    def __iter__(self) -> BaseVideoIterator:
        """
        :rtype: BaseVideoIterator
        """
        return self

    def init_cap(self):
        """
        Initialize the video file and read through the frames until the start frame is reached.
        """
        if self.video_start_index > 30:
            self.logger.info("Initializing video capture: skipping {} frames...".format(self.video_start_index))

        self.video.init_video_file_reader()

        # Iterate through the video frames until start point reached
        while self.current_video_index != self.video_start_index:
            self.video.read_next_frame(self.current_video_index)
            self.current_video_index += 1

    def close(self):
        """
        Close and release the resources of this video iterator.
        """
        self.video.close_video_file_reader()

    def read_next(self) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]], bool]:
        """
        :return: the next frame of the video loader with the corresponding label if it exists.
            Result: Optional[(frame_tensor, Optional[label_tensor], is_start_of_sequence)]
            frame_tensor: (batch_size, channels, height, width)
            label_tensor: (batch_size, height, width)
        :rtype: (torch.Tensor, torch.Tensor, bool)
        """
        if self.current_video_index >= self.video_start_index + self.frame_count:
            return None

        frame, label, is_start_of_sequence = self.video.read_next(self.current_video_index)
        self.current_video_index += 1

        return frame, label, is_start_of_sequence

    def __next__(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns the next frame of the video loader with the corresponding label if it exists.
        Result: (frame_tensor, Optional[label_tensor])
        frame_tensor:
            - (channels, height, width)
            - with RGB colors
            - with values in range 0..1
        label_tensor:
            - (height, width)
            - with values in range int64
        :return:
        :rtype: (torch.Tensor, torch.Tensor)
        """
        raise NotImplementedError

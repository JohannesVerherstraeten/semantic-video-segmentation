import logging
import torch
from typing import *


class BaseVideo(object):
    """
    Base class for video loaders of a single video.

    A video loader contains the metadata of a video.
    """

    def __init__(self, label_files: List[str], label_interval: int, video_start_index: int = 0,
                 video_end_index: int = -1, label_start_index: int = 0, logger: logging.Logger = None):
        """
        :param label_files: paths to the label files.
        :type label_files: list[str]
        :param label_interval: number of frames per label.
        :type label_interval: int
        :param video_start_index: frame index where the video starts.
        :type video_start_index: int
        :param video_end_index: the video stops at this frame index (not included).
        :type video_end_index: int
        :param label_start_index: frame index where the first label occurs.
        :type label_start_index: int
        """
        super(BaseVideo, self).__init__()

        self.logger = logger if logger is not None else logging.getLogger(self.__class__.__name__)

        self.label_files = label_files
        self.video_start_index = video_start_index
        self.video_end_index = video_end_index if video_end_index > -1 else self._get_total_nb_of_frames()
        self.label_start_index = label_start_index
        self.label_interval = label_interval

        self.frame_count = self.video_end_index - video_start_index
        self.label_count = len(self.label_files)

        if self.frame_count < 0:
            raise RuntimeError("Frame count must be positive. Got {}.".format(self.frame_count))

        if self.frame_count % self.label_interval != 0:
            self.logger.warning("Label interval {} is not a divisor of the number of frames {}."
                                .format(self.label_interval, self.frame_count))
            self.frame_count = self.frame_count - (self.frame_count % self.label_interval)
            self.logger.info("Reducing number of frames to {} by skipping the last ones.".format(self.frame_count))

        if self.frame_count < (self.label_interval * self.label_count):
            self.logger.warning("Too much labels for the number of frames in the video: got {}, expected {}."
                                .format(self.label_count, self.frame_count // self.label_interval))
            self.label_count = self.frame_count // self.label_interval
            self.logger.info("Reducing number of labels to {} by skipping the last ones.".format(self.label_count))

        if self.frame_count > (self.label_interval * self.label_count):
            self.logger.warning("Too much video frames for the number of labels: got {}, expected {}."
                                .format(self.frame_count, self.label_interval * self.label_count))
            self.frame_count = self.label_interval * self.label_count
            self.logger.info("Reducing number of frames to {} by skipping the last ones.".format(self.frame_count))

    def __len__(self) -> int:
        """
        :return: number of frames in this video.
        :rtype: int
        """
        return self.get_nb_images()

    def get_nb_images(self) -> int:
        """
        :return: The total number of frames in this video.
        :rtype: int
        """
        return self.frame_count

    def _get_total_nb_of_frames(self) -> int:
        """
        :return: the total number of frames, including the ones that will be skipped.
        :rtype: int
        """
        raise NotImplementedError

    def get_nb_labels(self) -> int:
        """
        :return: The total number of labels in this video.
        :rtype: int
        """
        return self.label_count

    def get_label_interval(self) -> int:
        """
        :return: the interval at which the frames are labeled. This must be constant across the whole video.
        :rtype: int
        """
        return self.label_interval

    def labels(self) -> Iterator[torch.Tensor]:
        """
        :return: an iterator over the labels in this dataset.
        :rtype: Iterator[torch.Tensor]
        """
        class LabelIterator(object):

            def __init__(self, video_file):
                """
                :param video_file:
                :type video_file: BaseVideo
                """
                self.video_file = video_file
                self.index = 0

            def __next__(self):
                """
                :return:
                :rtype: torch.Tensor
                """
                if self.index == self.video_file.get_nb_labels():
                    raise StopIteration
                else:
                    result = self.video_file.read_next_label(self.index)
                    self.index += 1
                    return result

            def __len__(self):
                return len(self.video_file)

            def __iter__(self):
                return self

        return LabelIterator(self)

    def init_video_file_reader(self):
        """
        :return:
        :rtype:
        """
        raise NotImplementedError

    def read_next(self, index: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        """
        Read next frame and label.

        Includes the is_start_of_sequence flag.

        Frame:
            - (channels, height, width)
            - with RGB colors
            - with values in range 0..1
        Label:
            - (height, width)
            - with values in range int64

        :param index: the index of the next frame. This function is guaranteed to be called with a frame index that
            is one more than the previous time, or with a frame index 0 after an initialize.
        :type index: int
        :return:
        :rtype: Tuple[torch.Tensor, Optional[torch.Tensor], bool]
        """
        return self.read_next_frame(index), self.read_next_label(index), index == self.video_start_index

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
        raise NotImplementedError

    def read_next_label(self, frame_idx: int) -> Optional[torch.Tensor]:
        """
        Label:
            - (height, width)
            - with values in range int64
        :param frame_idx: the index of the next frame. This function is guaranteed to be called with a label index that
            is one more than the previous time, or with a label index 0 after an initialize.
        :type frame_idx: int
        :return:
        :rtype: torch.Tensor
        """
        if (frame_idx - self.label_start_index) % self.label_interval == 0:
            label_index = (frame_idx - self.label_start_index) // self.label_interval
            return self.read_label_from_file(self.label_files[label_index])
        else:
            return None

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
        raise NotImplementedError

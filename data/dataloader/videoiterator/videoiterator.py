from typing import *
import torch

from .basevideoiterator import BaseVideoIterator
from data.dataset.datatype.video import Video


class VideoIterator(BaseVideoIterator):
    """
    Iterator over a video.
    """

    def __init__(self, video: Video):
        """
        :param video:
        :type video: Video
        """
        super(VideoIterator, self).__init__(video)
        self.init_cap()

    def __next__(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        """
        :return: the next frame of the video loader with the corresponding label if it exists.
            Result: (frame_tensor, Optional[label_tensor], is_start_of_sequence)
            frame_tensor: (batch_size, channels, height, width)
            label_tensor: (batch_size, height, width)
        :rtype: (torch.Tensor, torch.Tensor, bool)
        """
        result = self.read_next()
        if result is None:
            self.close()
            raise StopIteration
        else:
            return result

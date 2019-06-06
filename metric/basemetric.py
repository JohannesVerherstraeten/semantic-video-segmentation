from typing import *
from torch.tensor import Tensor

import logging


class BaseMetric(object):

    def __init__(self, keep_predictions: bool = False, keep_labels: bool = False, keep_frames: bool = False,
                 timesteps: int = 0, name: str = None):
        super(BaseMetric, self).__init__()
        self.name = name if name is not None else self.__class__.__name__
        self.logger = logging.getLogger(self.name)

        self.keep_predictions = keep_predictions
        self.keep_labels = keep_labels
        self.keep_frames = keep_frames
        self.timesteps = timesteps

        self.predictions = []
        self.labels = []
        self.frames = []

    def add(self, predictions: Optional[Tensor] = None, labels: Optional[Tensor] = None, frames: Optional[Tensor] = None) -> Tuple[Optional[float], Dict]:
        """
        Adds the predictions, labels and frames at the current timestep.
        Returns the metric value of the current timestep.
        """
        if predictions is not None:
            predictions = predictions.detach()
        if labels is not None:
            labels.detach()
        if frames is not None:
            frames.detach()
        self._store_current_input(predictions, labels, frames)
        return self._add(predictions, labels, frames)

    def _add(self, predictions: Optional[Tensor] = None, labels: Optional[Tensor] = None, frames: Optional[Tensor] = None) -> Tuple[Optional[float], Dict]:
        raise NotImplementedError

    def value(self) -> Tuple[Optional[float], Dict]:
        """
        Returns the accumulated metric value of all previous input data.
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the accumulated metric value of all input data and the metric state of the current sequence.

        Should happen at start of every epoch.
        """
        self.reset_state()

    def _store_current_input(self, predictions: Optional[Tensor] = None, labels: Optional[Tensor] = None, frames: Optional[Tensor] = None):
        """
        Adds the current input to the metric state of the current sequence.
        """
        if self.keep_predictions:
            self.__store_element(self.predictions, predictions.detach() if predictions is not None else None)
        if self.keep_labels:
            self.__store_element(self.labels, labels.detach() if labels is not None else None)
        if self.keep_frames:
            self.__store_element(self.frames, frames.detach() if frames is not None else None)

    def __store_element(self, lst: List, element: Optional[Tensor]):
        if self.timesteps == 0:
            return
        elif len(lst) == self.timesteps + 1:    # previous timestep + current timestep
            del lst[0]
            lst.append(element)
        else:
            lst.append(element)

    def get_previous_predictions(self) -> Optional[Tensor]:
        if len(self.predictions) < 2:
            return None
        else:
            return self.predictions[-2]

    def get_previous_labels(self) -> Optional[Tensor]:
        if len(self.labels) < 2:
            return None
        else:
            return self.labels[-2]

    def get_previous_frames(self) -> Optional[Tensor]:
        if len(self.frames) < 2:
            return None
        else:
            return self.frames[-2]

    def reset_state(self):
        """
        Resets the metric state of the current sequence.

        Should happen at start of every video sequence.
        """
        self.predictions = []
        self.labels = []
        self.frames = []

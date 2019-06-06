"""
Video files cannot be loaded using the default PyTorch dataloaders, because of
thread safety problems. The default dataloader will try to load frames concurrently,
which is not possible. This can be solved by making our own threaded videoiterator,
loading every video file in its own thread.

Current version supports multi threading and multiprocessing?
Not sure about correctness of the mutliprocessing. See notes on the _DataLoaderIter class
of PyTorch: torch.utils.data.dataloader._DataLoaderIter
This class should work similarly.

For multiprocessing, uncomment the lines related to the multiprocessing package
and comment out the lines related to the threading package.

Code based on PyTorch dataloader.py: torch.utils.data.dataloader
"""

import torch
import torch.utils.data.dataloader
import threading
from typing import *

from .basevideoiterator import BaseVideoIterator
from data.dataset.datatype.video import Video

# ---- multiprocessing ----
import sys
import traceback
import queue

# from torch._C import _update_worker_pids, _remove_worker_pids


class ExceptionWrapper(object):
    r"""Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
# ---- end multiprocessing ----


class ThreadedVideoIterator(BaseVideoIterator):
    """
    Iterator over a video file loader.
    """

    def __init__(self, video: Video, queue_size=2):
        """
        :param video:
        :type video: Video
        :param queue_size: how many video frames should be queued
        :type queue_size: int
        """
        super(ThreadedVideoIterator, self).__init__(video)

        self.queue_size = queue_size
        # self.index_queue = multiprocessing.SimpleQueue()            # Queue with the frame indices to load
        # self.worker_result_queue = multiprocessing.SimpleQueue()
        self.index_queue = queue.Queue()                            # Queue with the frame indices to load
        self.worker_result_queue = queue.Queue()
        self.frames_outstanding = 0                                 # how many frames are currently being loaded
        self.worker_pids_set = False
        self.shutdown = False

        # self.worker = multiprocessing.Process(target=self.worker_loop, name="Worker-" + self.name)
        self.worker = threading.Thread(target=self.worker_loop)
        self.worker.daemon = True                                  # ensure that the worker exits on process exit
        self.worker.start()

        # _update_worker_pids(id(self), self.worker.pid)
        # _set_SIGCHLD_handler()
        # self.worker_pids_set = True

        for _ in range(self.queue_size):
            self.request_next_frame()

    def request_next_frame(self):
        """
        Submit a request to the worker thread to load the next frame.
        """
        assert self.frames_outstanding < self.queue_size

        self.index_queue.put(True)
        self.frames_outstanding += 1

    def worker_loop(self):
        """
        Code being executed by the worker thread.
        """
        # TODO use shared storage?
        torch.set_num_threads(1)

        self.init_cap()

        watchdog = torch.utils.data.dataloader.ManagerWatchdog()

        while True:
            try:
                load_next = self.index_queue.get(timeout=5.0)
            except queue.Empty:
                if watchdog.is_alive():
                    continue
                else:
                    break
            if not load_next:       # worker has been shut down
                break
            try:
                result = self.read_next()
                if result is None:  # no more frames
                    break
                else:
                    frame, label, is_start_of_sequence = result
            except Exception:
                self.worker_result_queue.put(ExceptionWrapper(sys.exc_info()))
            else:
                self.worker_result_queue.put((frame, label, is_start_of_sequence))
                del frame, label    # ?
        self.worker_result_queue.put(False)

    def __next__(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        """
        :return: the next frame of the video loader with the corresponding label if it exists.
            Result: (frame_tensor, Optional[label_tensor])
            frame_tensor: (batch_size, channels, height, width)
            label_tensor: (batch_size, height, width)
        :rtype: (torch.Tensor, torch.Tensor)
        """
        if self.shutdown:
            raise StopIteration

        try:
            worker_result = self.worker_result_queue.get()
        except (FileNotFoundError, ImportError, queue.Empty):
            # Many weird errors can happen here due to Python
            # shutting down. These are more like obscure Python bugs.
            # FileNotFoundError can happen when we rebuild the fd
            # fetched from the queue but the socket is already closed
            # from the worker side.
            # ImportError can happen when the unpickler loads the
            # resource from `get`.
            self.shutdown_worker()
            raise StopIteration
        if not worker_result:
            self.shutdown_worker()
            raise StopIteration
        elif isinstance(worker_result, ExceptionWrapper):
            self.logger.critical("{}: {}".format(worker_result.exc_type, worker_result.exc_msg))
            self.shutdown_worker()
            raise StopIteration
        else:
            self.frames_outstanding -= 1
            self.request_next_frame()
            return worker_result

    def shutdown_worker(self):
        """
        Shutdown the worker thread.
        """
        if self.shutdown:
            return

        self.shutdown = True
        self.index_queue.put(False)

        # if some workers are waiting to put, make place for them
        try:
            while not self.worker_result_queue.empty():
                self.worker_result_queue.get()
        except (FileNotFoundError, ImportError):
            # Many weird errors can happen here due to Python
            # shutting down. These are more like obscure Python bugs.
            # FileNotFoundError can happen when we rebuild the fd
            # fetched from the queue but the socket is already closed
            # from the worker side.
            # ImportError can happen when the unpickler loads the
            # resource from `get`.
            pass
        self.worker_result_queue.put(False)

        # # removes pids no matter what
        # if self.worker_pids_set:
        #     # _remove_worker_pids(id(self))
        #     self.worker_pids_set = False

    def close(self):
        """
        Close and release the resources of this video iterator.
        """
        self.shutdown_worker()
        super(ThreadedVideoIterator, self).close()

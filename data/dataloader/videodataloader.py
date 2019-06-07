from __future__ import annotations
from data.dataloader.basedataloader import BaseDataLoader

import data.dataset.dataset.videodataset as videodataset
from data.dataloader.batchsampler.videobatchsampler import VideoBatchSampler
from .videoiterator.batchvideoiterator import BatchVideoIterator


class VideoDataLoader(BaseDataLoader):
    """
    Data loader for video datasets.

    Combines a dataset and a sampler.

    Iterating over this data loader returns elements of the form:
    (images, labels, is_start_of_sequence)
    with type
    (torch.Tensor, Torch.Tensor, bool)

    Image:
        - (batch_size, channels, height, width)
        - with RGB colors
        - with values in range 0..1
    Label:
        - (batch_size, height, width)
        - with values in range int64

    The is_start_of_sequence flag indicates whether the state of the network should be reset
    before processing this image. In this case the dataset consists of sequences of
    images, so the flag will be True for the first image of each sequence.
    """

    def __init__(self, dataset: videodataset.VideoDataset, batch_size=1, shuffle=False, num_workers=0):
        """
        :param dataset: dataset from which to load the data.
        :type dataset: VideoDataset
        :param batch_size: how many samples per batch to load.
        :type batch_size: int
        :param shuffle: set to ``True`` to have the data reshuffled at every epoch. Not (yet) compatible with
            batch_size > 1.
        :type shuffle: bool
        """
        super(VideoDataLoader, self).__init__()
        self.dataset = dataset
        self.use_workers = num_workers > 0
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_dataset(self) -> videodataset.VideoDataset:
        """
        Get the dataset of this dataloader.
        """
        return self.dataset

    def __iter__(self):
        return VideoDataLoaderIter(self)

    def __len__(self):
        return self.get_nb_images()

    def get_nb_images(self):
        return sum(video.get_nb_images() for video in self.dataset.get_videos())

    def get_nb_labels(self):
        return sum(video.get_nb_labels() for video in self.dataset.get_videos())


class VideoDataLoaderIter(object):

    def __init__(self, loader: VideoDataLoader):
        """
        :param loader:
        :type loader: VideoDataLoader
        """
        super(VideoDataLoaderIter, self).__init__()

        self.loader = loader
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.shuffle = loader.shuffle
        self.sampler = VideoBatchSampler(self.dataset, self.batch_size, self.shuffle)
        self.sampler_iteratator = iter(self.sampler)
        self.use_workers = loader.use_workers
        self.current_videoiter = None

    def __iter__(self):
        return self

    def __next__(self):
        # Iterate over the videos in the dataset one by one.
        if self.current_videoiter is None:
            video_idxs = next(self.sampler_iteratator)          # may raise StopIteration
            videos = [self.dataset.get_videos()[i] for i in video_idxs]
            self.current_videoiter = BatchVideoIterator(*videos, use_workers=self.use_workers)
        try:
            result = next(self.current_videoiter)
        except StopIteration:
            self.current_videoiter = None
            return next(self)       # try to open a new video iterator, if not available throw StopIteration
        else:
            assert len(result) == 3    # (frames, labels, flag)
            return result

    def __len__(self):
        raise NotImplementedError

    def get_nb_images(self):
        return self.dataset.get_nb_images()

    def get_nb_labels(self):
        return self.dataset.get_nb_labels()

"""
TODO clean up: make more readable

TODO Does not work yet with backpropagation through time > 0? Test this...
"""

import os
import shutil
import time
import logging
import datetime
import json
import torch
import torch.optim
import time
import gc
from math import inf
from torch.autograd import Variable
import torch.nn.utils.clip_grad
import model


import utils.datautils as datautils
import utils.transforms as transforms
import utils.utils as utils
import evaluate


def create_optimizer(model, config):
    # trainable_params = filter(lambda p: p.requires_grad, model.recurrent_parameters())
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = utils.get_instance(torch.optim, config["optimizer"], trainable_params)
    return optimizer


def create_lrscheduler(optimizer, config):
    lr_scheduler = utils.get_instance(torch.optim.lr_scheduler, config["lr_scheduler"], optimizer)
    return lr_scheduler


class Trainer(object):

    def __init__(self, model, data_loader_train, data_loader_val, loss_function, metrics, config_dict,
                 cuda, epochs, validation_freq, save_dir, resume_path=None,
                 max_bptt_depth=0, clip_grad_max=0.9, checkpoint_freq=10):
        """
        :param model:
        :type model: model.basemodel.BaseModel
        :param data_loader_train:
        :type data_loader_train: data.dataloader.basedataloader.BaseDataLoader
        :param data_loader_val:
        :type data_loader_val: data.dataloader.basedataloader.BaseDataLoader
        :param loss_function: the loss function.
        :type loss_function: metric.loss.baseloss.BaseLoss
        :param metrics:
        :type metrics: list[metric.metric.BaseMetric]
        :param config_dict:
        :type config_dict: dict
        :param cuda:
        :type cuda: bool
        :param epochs:
        :type epochs: int
        :param validation_freq:
        :type validation_freq: int
        :param save_dir:
        :type save_dir: str
        :param resume_path:
        :type resume_path: str
        :param max_bptt_depth: backpropagation through time: how many timesteps should be backpropagated in case of RNN.
        :type max_bptt_depth: int
        """
        self.config = config_dict
        self.logger = logging.getLogger(self.__class__.__name__)

        self.cuda = cuda
        self.model = model.cuda() if cuda else model

        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val

        self.loss_function = loss_function
        self.metrics = metrics
        self.eval_metric = metrics[0]

        self.best_filename = None
        self.best_loss = float("inf")
        self.best_eval_metric = -float("inf")  # TODO generalize
        self.current_val_loss = float("inf")
        self.current_val_eval_metric = -float("inf")

        self.start_epoch = 0
        self.current_epoch = 0

        self.target_epochs = epochs
        self.validation_freq = validation_freq
        self.checkpoint_freq = checkpoint_freq
        self.max_bptt_depth = max_bptt_depth
        self.clip_grad_max = float(clip_grad_max)

        start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.best_checkpoint_dir = os.path.join(save_dir, config_dict['name'], start_time)
        self.last_checkpoint_dir = os.path.join(save_dir, config_dict['name'], "last_checkpoint_path/")
        self.last_checkpoint_path = os.path.join(self.last_checkpoint_dir, "checkpoint.pth")
     
        if os.path.exists(self.last_checkpoint_path):
            self.resume_path = self.last_checkpoint_path
        else:
            self.resume_path = resume_path

        if self.resume_path:
            self.resume_checkpoint(self.resume_path)
        else:
            self.optimizer = create_optimizer(self.model, self.config)
            self.lr_scheduler = create_lrscheduler(self.optimizer, self.config)

    def train(self):
        t0 = time.time()

        datautils.ensure_dir(self.best_checkpoint_dir)
        datautils.ensure_dir(self.last_checkpoint_dir)
        train_log_save_path = os.path.join(self.best_checkpoint_dir, 'train.log')
        self.train_logger = TrainingLogger(train_log_save_path)

        self.train_logger.write("[Epoch: {}] Training phase started.".format(self.start_epoch))
        self.logger.info("[Epoch: {}] Training phase started.".format(self.start_epoch))

        # Save configuration file into checkpoint directory:
        config_save_path = os.path.join(self.best_checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=False)

        config_save_path = os.path.join(self.last_checkpoint_dir, 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=False)

        if self.resume_path:
            self.train_logger.write("Resuming from file: {}".format(self.resume_path))

        filename = None
        best_filename = None

        for epoch in range(self.start_epoch, self.target_epochs):
            self.current_epoch = epoch
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch=epoch)

            train_loss, train_eval_metric, train_metrics_log = self.train_epoch()

            t1 = time.time()
            self.train_logger.write("[Epoch: {0:d}] [Elapsed: {1:.0f}h{2:.0f}m{3:.0f}s] {{'Training loss': {4:.8f}, "
                                    "'Training evaluation metric': {5:.4f}, 'Training log:': {6} }}"
                                    .format(epoch, *transforms.seconds_to_hms(t1 - t0), train_loss, train_eval_metric,
                                            logToStr(train_metrics_log)))
            self.logger.info("[Epoch: {0:d}] [Elapsed: {1:.0f}h{2:.0f}m{3:.0f}s] Training loss: {4:.8f} | "
                             "Training metric: {5:.4f}"
                             .format(epoch, *transforms.seconds_to_hms(t1 - t0), train_loss, train_eval_metric))

            # Validation
            best_checkpoint_created = False
            if (epoch + 1) % self.validation_freq == 0 or epoch + 1 == self.target_epochs:
                self.current_val_loss, self.current_val_eval_metric, current_val_metrics_log = self.validate_epoch()

                t2 = time.time()
                self.train_logger.write("[Epoch: {0:d}] [Elapsed: {1:.0f}h{2:.0f}m{3:.0f}s] {{'Validation loss': {4:.8f}, "
                                        "'Validation metric': {5:.4f}, 'Validation log': {6} }}"
                                        .format(epoch, *transforms.seconds_to_hms(t2 - t0), self.current_val_loss,
                                                self.current_val_eval_metric, logToStr(current_val_metrics_log)))

                self.logger.info("[Epoch: {0:d}] [Elapsed: {1:.0f}h{2:.0f}m{3:.0f}s] Validation loss: {4:.8f} | "
                                 "Validation metric: {5:.4f}"
                                 .format(epoch, *transforms.seconds_to_hms(t2 - t0), self.current_val_loss,
                                         self.current_val_eval_metric))

                # Save model if it's the best thus far
                if self.current_val_eval_metric > self.best_eval_metric or self.current_val_loss < self.best_loss:
                    if self.current_val_eval_metric > self.best_eval_metric:
                        self.best_eval_metric = self.current_val_eval_metric
                        self.best_filename = 'selftrained-checkpoint-e{}.pth'.format(self.current_epoch)
                    if self.current_val_loss < self.best_loss:
                        self.best_loss = self.current_val_loss
                    best_filename = 'selftrained-checkpoint-e{}.pth'.format(self.current_epoch)
                    checkpoint_path = os.path.join(self.best_checkpoint_dir, best_filename)
                    self.save_checkpoint(checkpoint_path)
                    self.train_logger.write("Checkpoint created. Best validation accuracy / loss so far: {}"
                                            .format(self.current_val_eval_metric))
                    self.logger.info("Checkpoint created. Best validation accuracy / loss so far: {}"
                                     .format(self.current_val_eval_metric))
                    best_checkpoint_created = True

            # create checkpoint for resuming in case of interrupted job
            self.save_checkpoint(self.last_checkpoint_path)

            # create checkpoint regularly
            if not best_checkpoint_created and (epoch + 1) % self.checkpoint_freq == 0:
                filename = 'selftrained-checkpoint-regular-e{}.pth'.format(self.current_epoch)
                checkpoint_path = os.path.join(self.best_checkpoint_dir, filename)
                self.save_checkpoint(checkpoint_path)

        t3 = time.time()
        self.train_logger.write("Training finished. Elapsed time: {0:.0f}h{1:.0f}m{2:.0f}s"
                                .format(*transforms.seconds_to_hms(t3 - t0)))
        self.logger.info("Training finished. Elapsed time: {0:.0f}h{1:.0f}m{2:.0f}s"
                         .format(*transforms.seconds_to_hms(t3 - t0)))

        shutil.rmtree(self.last_checkpoint_dir)

        if self.best_filename is not None:
            checkpoint_path = os.path.join(self.best_checkpoint_dir, self.best_filename)
            self.logger.info("Loading best checkpoint: {}".format(checkpoint_path))
            self.resume_checkpoint(checkpoint_path)

            self.config["evaluator"]["archs"] = [checkpoint_path]
            evaluate.main(self.config["evaluator"], False, self.cuda, False, True)

            self.logger.info("Done")

    def train_epoch(self):
        """
        :return: (train_loss, train_evaluation_metric, train_metrics_log)
        :rtype: (float, float, Any)
        """
        return self.__eval_epoch(self.data_loader_train, backpropagate=True)

    def validate_epoch(self):
        """
        :return: (validation_loss, validation_evaluation_metric, validation_metrics_log)
        :rtype: (float, float, Any)
        """
        return self.__eval_epoch(self.data_loader_val, backpropagate=False)

    def __eval_epoch(self, data_loader, backpropagate):
        """
        :param data_loader:
        :type data_loader: data.dataloader.basedataloader.BaseDataLoader
        :param backpropagate:
        :type backpropagate: bool
        :return: (loss, metric, metrics_log)
        :rtype: (float, float, Any)
        """

        # Enable / disable dropout etc...
        if backpropagate:
            self.model.train()
        else:
            self.model.eval()

        # Do a full reset of the metrics and loss to clear previous epoch data.
        [metric.reset() for metric in self.metrics]
        self.loss_function.reset()

        # Iterate over data loader
        # - inputs: (batch_size, channels, height, width)
        # - labels: (batch_size, height, width). Can be None in case of video files, but not in case of independent
        #       images.
        # - is_start_of_video: bool indicating that all input frames from the batch are the first frame of new video
        #       file. This triggers the reset of the RNN states. Is always True in case of independent images.
        #       Added to make abstraction of image / video data.
        for i, (inputs, labels, is_start_of_video) in enumerate(data_loader):

            # Move inputs and labels to cuda if needed
            inputs, labels = self.__preprocess_inputs(inputs, labels)

            # Start of video file: reset the sequence state of the metric loss and model
            if is_start_of_video:
                [metric.reset_state() for metric in self.metrics]
                self.loss_function.reset_state()    # temporal loss functions may have a state
                self.model.reset_state()            # RNNs have a state

            if backpropagate and (labels is None or self.max_bptt_depth == 0):
                # TODO: currently, only bptt >= 1 is supported.
                # This block is not executed in evaluation mode (no backpropagation) or
                # in training mode when a label is available (and thus the loss will be calculated) AND max_bptt == 1.
                self.model.repackage_hidden_state()
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(backpropagate):

                # Forward propagation
                # outputs: (batch_size, nb_classes, height, width)
                outputs = self.model(inputs)

                # Calculate loss and metrics
                loss, loss_log = self.loss_function.loss(predictions=outputs,
                                                         labels=labels,
                                                         frames=inputs)
                [metric.add(outputs, labels) for metric in self.metrics]

                if backpropagate:
                    # Backpropagate loss
                    if loss is not None:
                        loss.backward()

                        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                        total_norm = torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.clip_grad_max,
                                                                              norm_type=2)
                        self.optimizer.step(None)

        total_loss, total_loss_log = self.loss_function.value()
        total_loss /= data_loader.dataset.get_nb_labels()
        eval_metric_value, _ = self.metrics[0].value()      # The first metric is the evaluation metric.
        metrics_log = [metric.value()[1] for metric in self.metrics]
        total_log = {"loss": total_loss_log, "metrics": metrics_log}
        total_log["nb_of_images"] = data_loader.get_nb_images()
        total_log["nb_of_labels"] = data_loader.get_nb_labels()

        return total_loss, eval_metric_value, total_log

    def __preprocess_inputs(self, inputs, labels):
        inputs = Variable(inputs)
        if self.cuda:
            inputs = inputs.cuda()
        if labels is not None:
            labels = Variable(labels)
            if self.cuda:
                labels = labels.cuda()
        return inputs, labels

    def save_checkpoint(self, save_path):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': self.current_epoch,
            # 'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.current_val_loss,
            'eval_metric': self.current_val_eval_metric,
            # 'monitor_best': self.monitor_best,
            'config': self.config,
            'best_filename': self.best_filename,
            'best_eval_metric': self.best_eval_metric
        }
        self.logger.info("Saving checkpoint: {} ...".format(save_path))
        torch.save(state, save_path)

    def resume_checkpoint(self, resume_path):
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu" if not self.cuda else None)

        self.current_epoch = checkpoint['epoch']
        self.start_epoch = checkpoint['epoch'] + 1
        # self.monitor_best = checkpoint['monitor_best']
        # self.train_logger = checkpoint['logger']
        # self.best_loss = checkpoint['loss']

        if 'eval_metric' not in checkpoint:
            # Backward compatibility
            self.logger.info("Old checkpoint type detected. No warnings not guaranteed. ")
            # self.best_eval_metric = checkpoint['miou']

            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            # self.best_eval_metric = checkpoint['eval_metric']

            # load architecture params from checkpoint.
            if checkpoint['config']['arch'] != self.config['arch']:
                self.logger.warning(
                    'Warning: Architecture configuration given in config file is different from that of checkpoint. ' +
                    'This may yield an exception while state_dict is being loaded.')
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)

            # load optimizer state from checkpoint only when optimizer type is not changed.
            if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
                self.logger.warning('Warning: Optimizer type given in config file is different from that of checkpoint. ' + \
                                    'Optimizer parameters not being resumed.')
                self.optimizer = create_optimizer(self.model, self.config)
            else:
                try:
                    self.optimizer = create_optimizer(self.model, self.config)
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                except ValueError as e:
                    self.logger.warning("{}. Optimizer not loaded from checkpoint...".format(e))
                    self.optimizer = create_optimizer(self.model, self.config)
            self.lr_scheduler = create_lrscheduler(self.optimizer, self.config)

            if 'best_filename' in checkpoint.keys():
                self.best_filename = checkpoint['best_filename']
                self.best_eval_metric = checkpoint['best_eval_metric']
            else:
                self.best_filename = None
                self.best_eval_metric = -float("inf")

        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))


class TrainingLogger(object):
    """
    Training process logger
    Note:
        Used by BaseTrainer to save training history.
    """

    def __init__(self, logfile):
        self.logfile = logfile

    def write(self, string):
        with open(self.logfile, 'a') as summary_file:
                        summary_file.write(string + "\n")

                
def logToStr(log_dict):
    if isinstance(log_dict, dict):
        return dict([(str(key), logToStr(val)) for key, val in log_dict.items()])
    else:
        return str(log_dict)

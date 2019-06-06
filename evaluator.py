"""
TODO clean up: make more readable (same with trainer.py)
"""

import logging
import torch
from torch.nn.modules import BatchNorm2d
from torch.autograd import Variable
import os.path
import matplotlib.pyplot as plt
import utils.datautils as datautils
import json
import time
import numpy as np

from utils.visualize import Visualizer
import utils.opticalflow as opticalflow


class Evaluator(object):

    def __init__(self, data_loader, models=[], metrics=[], config=None, visualize=True, show_labels=True,
                 overlay=True, show_flow=False, save_dir=None, log_files=[], cuda=False):

        self.logger = logging.getLogger(self.__class__.__name__)

        self.data_loader = data_loader
        self.models = models
        self.metrics = metrics
        self.config = config
        self.cuda = cuda

        # Following attributes are only used when visualize=True.
        self.show_labels = show_labels
        self.show_flow = show_flow
        self.save_dir = save_dir
        self.eval_loggers = [EvaluationLogger(log_file) for log_file in log_files]
        self.interval = 0.02

        self.count = 0

        self.visualizer = None
        if visualize:
            assert not (show_flow and show_labels)
            self.visualizer = Visualizer(models, show_flow=show_flow, show_labels=show_labels, overlay=overlay,
                                         window_size=(15, 7))

    def evaluate(self):
        self.logger.info("Starting evaluation...")

        for model in self.models:
            model.eval()
        # for module in self.models[1].modules():
        #     if isinstance(module, BatchNorm2d):
        #         module.train()

        # self.models[1].train()
        # self.models[1].lstm.cells[0].print_parameters()
        # self.models[1].reset_trainable_parameters()

        [[metric.reset() for metric in model_metrics] for model_metrics in self.metrics]
        previous_frames = None

        for frames, labels, is_start_of_video in iter(self.data_loader):

            # Move inputs and labels to cuda if needed
            frames, labels = self.__preprocess_inputs(frames, labels)

            self.logger.info("{} / {}".format(self.count, self.data_loader.get_nb_images()))

            if is_start_of_video:
                [[metric.reset_state() for metric in model_metrics] for model_metrics in self.metrics]
                [model.reset_state() for model in self.models]
                previous_frames = None
                t5_new = time.time()
                
                # print("start of sequence")

            # tracker.print_diff()
            # gc.collect()
            # print(gc.get_count())
            # print(gc.garbage[-9:-1])
            # time.sleep(3)

            with torch.set_grad_enabled(False):

                predictions = []

                if len(self.models) > 0:

                    all_metric_values = []
                    all_metric_logs = []
                    all_metric_infos = []

                    for i, model in enumerate(self.models):
                        t0 = time.time()
                        output = model(frames)
                        _, prediction = torch.max(output.data, 1)
                        t1 = time.time()
                        predictions.append(prediction)

                        metric_values = []
                        metric_logs = []
                        metric_info = "Model {}: \t".format(i)
                        eval_logfile_info = dict()
                        for metric in self.metrics[i]:
                            metric_value, metric_log = metric.add(output, labels, frames)

                            if metric_value is None:    # not pretty
                                metric_info += "{}: {}   ".format(metric.name, metric_value)
                            else:
                                metric_info += "{}: {:.5f}   ".format(metric.name, metric_value)

                            eval_logfile_info[metric.name] = {"value": metric_value, "log": metric_log}

                        all_metric_values.append(metric_values)
                        all_metric_logs.append(metric_logs)
                        all_metric_infos.append(metric_info)

                        if len(self.eval_loggers) > i and self.eval_loggers[i] is not None:
                            self.eval_loggers[i].write("[{}][{}] {}".format(self.count, t1-t0, logToStr(eval_logfile_info)))

                    [self.logger.info(all_metric_info) for all_metric_info in all_metric_infos]
                    self.logger.info("")

                elif self.show_labels:
                    predictions.append(labels)

                if self.visualizer is not None:

                    save_path = None
                    if self.save_dir is not None:
                        datautils.ensure_dir(self.save_dir)

                        # Save configuration file into checkpoint directory:
                        config_save_path = os.path.join(self.save_dir, 'config.json')
                        with open(config_save_path, 'w') as handle:
                            json.dump(self.config, handle, indent=4, sort_keys=False)

                        save_path = os.path.join(self.save_dir, "fig_{}.png".format(self.count))

                    if self.show_labels:
                        self.visualizer.imshow(frames, predictions, class_encoding=self.data_loader.get_color_encoding(),
                                               save_path=save_path)

                    elif self.show_flow and previous_frames is not None:
                        flow_fw, flow_bw = opticalflow.optical_flow_2(previous_frames, frames)
                        warped, _ = opticalflow.warp_flow(previous_frames, flow_bw)

                        warp_error_1 = torch.exp(-50 * torch.pow(torch.norm(frames - warped, dim=1, keepdim=True), 2))
                        consistency_map = opticalflow.forward_backward_consistency(flow_fw, flow_bw, ref_frame_a=False)

                        self.visualizer.imshow_flow(previous_frames, frames, flow_bw, flow_fw, warped, warp_error_1,
                                                    consistency_map)
                    previous_frames = frames

                    if self.show_labels or self.show_flow:
                        plt.pause(0.03)
                    else:
                        plt.pause(self.interval)

            self.count += 1
            del predictions
            del labels
            t5 = time.time()

        t8 = time.time()

        self.logger.info("========================================")
        for eval_logger in self.eval_loggers:
            if eval_logger is not None:
                eval_logger.write("======================================")
        for j, model_metrics in enumerate(self.metrics):
            for metric in model_metrics:
                self.logger.info("{}: {}".format(metric.name, metric.value()))
                if len(self.eval_loggers) > j and self.eval_loggers[j] is not None:
                    self.eval_loggers[j].write("{}: {}".format(metric.name, metric.value()))

    def __preprocess_inputs(self, inputs, labels):
        if self.cuda:
            inputs = inputs.cuda()
        if labels is not None and self.cuda:
            labels = labels.cuda()
        return inputs, labels


class EvaluationLogger(object):
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

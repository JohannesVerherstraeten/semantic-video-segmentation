"""
TODO building the network from json file using software design pattern: factory, builder? -> will allow more freedom
 (easy composite loss function creation?), cleaner code, less code duplication.
"""

import sys
import logging
import json
import torch
import argparse
import os.path
import time

import data.weighing
import data.dataset
import data.dataloader
from evaluator import Evaluator
import metric
import metric.loss
import utils.utils as utils
import model


def main(config, visualize, visualization_interval, use_cuda, save_visualization, log):
    logging.basicConfig(level=logging.INFO,
                        # filename="evaluate.log",
                        stream=sys.stdout,
                        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                        datefmt='%d/%m/%Y %I:%M:%S %p')

    logger = logging.getLogger("Main")
    logger.info("===================== New run ==================================")

    # setup data_loader instances
    logger.info("Setting up dataset and data loader instances...")
    dataset_train, dataset_val, dataset_test = utils.get_dataset_instances(data.dataset, config["dataset"])
    dataloader_train = dataset_train.create_dataloader(**config["data_loader"]["train"]["kwargs"])
    dataloader_val = dataset_val.create_dataloader(**config["data_loader"]["val"]["kwargs"])
    dataloader_test = dataset_test.create_dataloader(**config["data_loader"]["test"]["kwargs"])
    nb_of_classes = dataset_train.get_nb_classes()

    # build model architecture
    model_instances = []
    log_files = []
    for model_description in config["archs"]:
        if isinstance(model_description, str):
            logger.info("Setting up model {}".format(model_description))
            model_instance = load_model_from_checkpoint(model_description, use_cuda)
            model_instance = model_instance.cuda() if use_cuda else model_instance
            model_instances.append(model_instance)
            if log:
                log_files.append(model_description[:-4] + "-evaluation.log")

        else:
            logger.info("Setting up model...")
            model_instance = utils.get_instance(model, model_description,
                                                num_classes=nb_of_classes)
            load_model_from_checkpoint(model_description["resume"], use_cuda, model_instance)
            model_instances.append(model_instance)
            if log:
                logger.warning("not logging to evaluation.log file.")
                log_files.append(None)

    # get loss functions
    conceptual_loss_weights = None
    if "conceptual_loss" in config:
        logger.info("Calculating class weights...")
        conceptual_loss_weights = utils.get_instance(data.weighing, config["conceptual_loss"]["weighing"],
                                                     dataset=dataset_train)
        conceptual_loss_weights = conceptual_loss_weights.cuda() if use_cuda else conceptual_loss_weights

    metrics = []
    for i in range(len(model_instances)):
        model_metrics = []
        if "conceptual_loss" in config:
            logger.info("Creating loss function...")
            conceptual_loss = utils.get_instance(metric.loss, config["conceptual_loss"],
                                                 class_weights=conceptual_loss_weights)
            model_metrics.append(conceptual_loss)
        if "consistency_loss" in config:
            # Use a new instance of the conceptual loss in the consistency loss function to avoid interference!
            conceptual_loss_2 = None
            if "conceptual_loss" in config:
                conceptual_loss_2 = utils.get_instance(metric.loss, config["conceptual_loss"],
                                                       class_weights=None,
                                                       weight=1.)
            consistency_loss = utils.get_instance(metric.loss, config["consistency_loss"],
                                                  conceptual_loss=conceptual_loss_2,
                                                  only_when_label=False)
            model_metrics.append(consistency_loss)
        if "metrics" in config:
            logger.info("Creating metrics...")
            dataset_labels = list(dataset_train.get_color_encoding())
            for metric_config in config["metrics"]:
                ignore_indices = ()
                if "ignore_labels" in metric_config["kwargs"]:   # TODO can be done cleaner: ignore indices in config immediately?
                    ignore_indices = [dataset_labels.index(label) for label in metric_config["kwargs"]["ignore_labels"]]
                    ignore_indices = tuple(ignore_indices)
                metric_i = utils.get_instance(metric, metric_config, num_classes=nb_of_classes,
                                              ignore_indices=ignore_indices)
                model_metrics.append(metric_i)

        metrics.append(model_metrics)

    save_dir = "saved/images/ " + config["name"] if save_visualization else None

    evaluator = Evaluator(dataloader_val, model_instances, metrics, config,
                          visualize, show_labels=True, overlay=True, show_flow=False,
                          save_dir=save_dir, log_files=log_files, cuda=use_cuda, interval=visualization_interval)
    evaluator.evaluate()


def load_model_from_checkpoint(checkpoint_path, cuda, model_instance=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu" if not cuda else None)
    if model_instance is None:
        model_instance = utils.get_instance(model, checkpoint["config"]["arch"])
    model_instance.load_state_dict(checkpoint['state_dict'], strict=False)
    return model_instance


def __restricted_float(x):
    if x is None:
        x = 0.0
    else:
        x = float(x)
    if x < 0.0:
        raise argparse.ArgumentTypeError("time parameter value {} must be non-negative".format(x))
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize one or multiple models')
    parser.add_argument('config', type=str,
                        help='config file path.')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='visualize the evaluation process.')
    parser.add_argument('-s', '--save', action='store_true',
                        help='save the visualization under saved/images/<config-name>/.')
    parser.add_argument('-t', '--time', type=__restricted_float,
                        help='time between consecutive visualizations (1 / framerate).')
    parser.add_argument('-l', '--log', action='store_true',
                        help='log the evaluation process to the same direction as the loaded checkpoint(s).')
    parser.add_argument('--cuda', action='store_true',
                        help="enable GPU acceleration.")

    args = parser.parse_args()

    if args.save and not args.visualize:
        print("Cannot save images if visualize flag -v is off.")

    config = json.load(open(args.config))

    main(config, args.visualize, args.time, args.cuda, args.save, args.log)

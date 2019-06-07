"""
TODO building the network from json file using software design pattern: factory, builder? -> will allow more freedom
 (easy composite loss function creation?) and cleaner code.

TODO allow checkpoint loading without creation of Trainer instance?

"""

import sys
import logging
import json
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import os.path
import time

import data.weighing
import data.dataset
import data.dataloader
import model
import trainer
import metric
import metric.loss
import utils.utils as utils
import utils.visualize as visualize
import utils.transforms as transforms
import evaluate


def main(config, checkpoint, use_cuda):
    logging.basicConfig(level=logging.INFO,
                        # filename="train.log",
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
    logger.info("Setting up model...")
    model_instance: model.BaseModel = utils.get_instance(model, config["arch"],
                                                         num_classes=nb_of_classes)

    # get loss functions TODO can be simplified with new metric - loss interface?
    logger.info("Calculating class weights...")
    conceptual_loss_weights = utils.get_instance(data.weighing, config["conceptual_loss"]["weighing"],
                                                 dataset=dataset_train)
    conceptual_loss_weights = conceptual_loss_weights.cuda() if use_cuda else conceptual_loss_weights

    logger.info("Creating loss function...")
    conceptual_loss = utils.get_instance(metric.loss, config["conceptual_loss"],
                                         class_weights=conceptual_loss_weights)
    if "consistency_loss" in config:
        # Use a new instance of the conceptual loss in the consistency loss function to avoid interference!
        conceptual_loss_2 = utils.get_instance(metric.loss, config["conceptual_loss"],
                                               class_weights=None,
                                               weight=1.)
        consistency_loss = utils.get_instance(metric.loss, config["consistency_loss"],
                                              conceptual_loss=conceptual_loss_2,
                                              only_when_label=True)
        loss = metric.loss.CompositeLoss(conceptual_loss, consistency_loss)
    else:
        loss = conceptual_loss

    # get metrics
    logger.info("Creating metrics...")
    metrics = []
    dataset_labels = list(dataset_train.get_color_encoding())
    for metric_config in config["metrics"]:
        ignore_indices = ()
        if "ignore_labels" in metric_config["kwargs"]:  # TODO can be done cleaner: ignore indices in config immediately?
            ignore_indices = [dataset_labels.index(label) for label in metric_config["kwargs"]["ignore_labels"]]
        metric_i = utils.get_instance(metric, metric_config, num_classes=nb_of_classes, ignore_indices=ignore_indices)
        metrics.append(metric_i)

    logger.info("Setting up trainer...")
    trainer_instance = trainer.Trainer(**config["trainer"]["kwargs"],
                                       model=model_instance,
                                       data_loader_train=dataloader_train,
                                       data_loader_val=dataloader_val,
                                       loss_function=loss,
                                       metrics=metrics,
                                       config_dict=config,
                                       cuda=use_cuda,
                                       resume_path=checkpoint)
    trainer_instance.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-e', '--evaluateonly', action='store_true',
                        help="don't train, only evaluate the checkpoint given in the evaluator log.")
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true', help="enable GPU acceleration")
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="disable GPU acceleration (default)")
    parser.set_defaults(cuda=False)

    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.evaluateonly:
        evaluate.main(config["evaluator"], False, args.cuda, False, True)
    else:
        main(config, args.resume, args.cuda)

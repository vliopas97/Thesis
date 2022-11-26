"""Utility functionns for model building, training and evaluation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import config
import argparse


def ssd_parser():
    """Instatiate a command line parser for ssd network model
    building, training, and testing
    """
    parser = argparse.ArgumentParser(description='SSD for object detection')
    # arguments for model building and training
    help_ = "Batch size during training"
    parser.add_argument("--batch_size",
                        default=8,
                        type=int,
                        help=help_)
    help_ = "Number of epochs to train"
    parser.add_argument("--epochs",
                        default=100,
                        type=int,
                        help=help_)
    help_ = "Specify the desired training steps for each epoch"
    parser.add_argument("--steps",
                        default=1000,
                        type=int,
                        help=help_)
    help_ = "Specify the desired size for the validation dataset"
    parser.add_argument("--val_size",
                        default=500,
                        type=int,
                        help=help_)
    help_ = "Epoch at which to start training (useful for resuming previous training run"
    parser.add_argument("--initial_epoch",
                        default=0,
                        type=int,
                        help=help_)
    help_ = "Number of data generator worker threads"
    parser.add_argument("--workers",
                        default=4,
                        type=int,
                        help=help_)
    help_ = "Labels IoU threshold"
    parser.add_argument("--threshold",
                        default=0.4,
                        type=float,
                        help=help_)
    help_ = "Train the model"
    parser.add_argument("--train",
                        action='store_true',
                        help=help_)
    help_ = "Print model summary (text and png)"
    parser.add_argument("--summary",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Directory for saving filenames"
    parser.add_argument("--save_dir",
                        default="weights",
                        help=help_)
    help_ = "Validate during training or not"
    parser.add_argument("--validate",
                        default=True,
                        action='store_true',
                        help=help_)
    help_ = "Configurations file location"
    parser.add_argument("--config",
                        default="configs/",
                        help=help_)
    help_ = "Learning rate used in training"
    parser.add_argument("--learning_rate",
                        default=10e-3,
                        type=float,
                        help=help_)

    # dataset configurations
    help_ = "Path to dataset directory"
    parser.add_argument("--data_path",
                        default="C:/coco",
                        help=help_)
    help_ = "Train labels csv file name"
    parser.add_argument("--train_set",
                        default="train2017",
                        help=help_)
    help_ = "Test labels csv file name"
    parser.add_argument("--val_set",
                        default="val2017",
                        help=help_)

    # configurations for evaluation of a trained model
    help_ = "Load h5 model trained weights"
    parser.add_argument("--restore_weights",
                        help=help_)
    help_ = "Evaluate model"
    parser.add_argument("--evaluate",
                        default=False,
                        action='store_true',
                        help=help_)
    help_ = "Image file for evaluation"
    parser.add_argument("--image_file",
                        default=None,
                        help=help_)
    help_ = "Class probability threshold (>= is an object)"
    parser.add_argument("--confidence_threshold",
                        default=0.5,
                        type=float,
                        help=help_)

    # debug configuration
    help_ = "Level of verbosity for print function"
    parser.add_argument("--verbose",
                        default=1,
                        type=int,
                        help=help_)

    return parser

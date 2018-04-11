from __future__ import print_function

import os
import sys
from argparse import ArgumentParser
import random
import logging
from importlib import util

import numpy as np
import pandas as pd

import torch
from torch.nn import Module
from torchvision.transforms import ToTensor

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engines import Events, Engine
from ignite.handlers import Timer
from ignite._utils import to_variable, to_tensor

from dataflow import get_test_data_loader
from common import setup_logger, save_conf


def write_model_graph(writer, model, cuda):
    try:
        dummy_input = to_variable(torch.rand(10, 3, 64, 64), cuda=cuda)
        writer.add_graph(model, dummy_input)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))


def create_inferencer(model, cuda=True):

    def _prepare_batch(batch):
        x, path = batch
        x = to_variable(x, cuda=cuda)
        return x, path

    def _update(engine, batch):
        x, files = _prepare_batch(batch)
        y_pred = model(x)
        return {
            "x": to_tensor(x, cpu=True),
            "y_pred": to_tensor(y_pred, cpu=True),
            "files": files
        }

    model.eval()
    inferencer = Engine(_update)
    return inferencer


def load_config(config_filepath):
    assert os.path.exists(config_filepath), "Configuration file '{}' is not found".format(config_filepath)
    # Handle local modules
    sys.path.insert(0, os.path.dirname(__file__))
    # Load custom module
    spec = util.spec_from_file_location("config", config_filepath)
    custom_module = util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    config = custom_module.__dict__
    assert "DATASET_PATH" in config, "DATASET_PATH parameter is not found in configuration file"
    assert os.path.exists(config["DATASET_PATH"]), \
        "Dataset '{}' is not found".format(config["DATASET_PATH"]) + \
        "Dataset can be downloaded from : http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    assert "OUTPUT_PATH" in config, "OUTPUT_PATH is not found in the configuration file"

    if "N_TTA" not in config:
        config["N_TTA"] = 5

    assert "MODEL" in config, "MODEL is not found in configuration file"
    assert isinstance(config["MODEL"], str) and os.path.isfile(config["MODEL"]), \
        "Model is not found at '{}'".format(config["MODEL"])
    config["MODEL"] = torch.load(config["MODEL"])
    assert isinstance(config["MODEL"], Module), \
        "Model should be an instance of torch.nn.Module, but given {}".format(type(config["MODEL"]))

    if "TEST_TRANSFORMS" not in config:
        config["TEST_TRANSFORMS"] = [ToTensor(), ]

    return config


def write_submission(files, y_preds, classes, submission_filepath):
    y_text = [classes[y] for y in y_preds]
    files = [os.path.basename(f) for f in files]
    df = pd.DataFrame(index=files, data=y_text)
    df.to_csv(submission_filepath, sep=' ', header=False)


def run(config_file):

    print("--- Tiny ImageNet 200 Playground : Inference --- ")

    print("Load config file ... ")
    config = load_config(config_file)

    seed = config.get("SEED", 2018)
    random.seed(seed)
    torch.manual_seed(seed)

    output = config["OUTPUT_PATH"]
    model = config["MODEL"]
    debug = config.get("DEBUG", False)

    from datetime import datetime
    now = datetime.now()
    log_dir = os.path.join(output, "inference_%s" % (now.strftime("%Y%m%d_%H%M")))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
        print("Activated debug mode")

    logger = logging.getLogger("Tiny ImageNet 200: Inference")
    setup_logger(logger, os.path.join(log_dir, "test.log"), log_level)

    logger.debug("Setup tensorboard writer")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    save_conf(config_file, logger, writer)

    cuda = torch.cuda.is_available()
    if cuda:
        logger.debug("CUDA is enabled")
        from torch.backends import cudnn
        cudnn.benchmark = True
        model = model.cuda()

    write_model_graph(writer, model=model, cuda=cuda)

    logger.debug("Setup test dataloader")
    dataset_path = config["DATASET_PATH"]
    test_data_transform = config["TEST_TRANSFORMS"]
    batch_size = config.get("BATCH_SIZE", 64)
    num_workers = config.get("NUM_WORKERS", 8)
    test_loader = get_test_data_loader(dataset_path, test_data_transform, batch_size, num_workers, cuda=cuda)

    logger.debug("Setup ignite trainer and evaluator")
    inferencer = create_inferencer(model, cuda=cuda)

    n_tta = config["N_TTA"]

    logger.debug("Setup handlers")
    # Setup timer to measure evaluation time
    timer = Timer(average=True)
    timer.attach(inferencer,
                 start=Events.STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED)

    n_samples = len(test_loader.dataset)
    files = np.zeros((n_samples, ), dtype=np.object)
    y_probas_tta = np.zeros((n_samples, 200, n_tta))

    @inferencer.on(Events.EPOCH_COMPLETED)
    def log_tta(engine):
        logger.debug("TTA {} / {}".format(engine.state.epoch, n_tta))

    @inferencer.on(Events.ITERATION_COMPLETED)
    def save_results(engine):
        output = engine.state.output
        tta_index = engine.state.epoch - 1
        start_index = ((engine.state.iteration - 1) % len(test_loader)) * batch_size
        end_index = min(start_index + batch_size, n_samples)
        batch_y_probas = output['y_pred'].numpy()
        y_probas_tta[start_index:end_index, :, tta_index] = batch_y_probas
        if tta_index == 0:
            files[start_index:end_index] = output['files']

    logger.debug("Start inference")
    try:
        inferencer.run(test_loader, max_epochs=n_tta)
    except KeyboardInterrupt:
        logger.info("Catched KeyboardInterrupt -> exit")
    except Exception as e:  # noqa
        logger.exception("")
        if debug:
            try:
                # open an ipython shell if possible
                import IPython
                IPython.embed()  # noqa
            except ImportError:
                print("Failed to start IPython console")
    writer.close()

    # Average probabilities:
    y_probas = np.mean(y_probas_tta, axis=-1)
    y_preds = np.argmax(y_probas, axis=-1)

    logger.info("Write submission file")
    submission_filepath = os.path.join(log_dir, "predictions.csv")
    write_submission(files, y_preds, test_loader.dataset.classes, submission_filepath)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("config_file", type=str, help="Configuration file. See examples in configs/")
    args = parser.parse_args()
    run(args.config_file)

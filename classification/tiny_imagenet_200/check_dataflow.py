from __future__ import print_function, division

import os
import sys
from argparse import ArgumentParser
import random
import logging
from importlib import util

import numpy as np
import pandas as pd
# Change matplotlib backend
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

import torch

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engines import Events, Engine
from ignite.handlers import Timer


def setup_logger(logger, output, level=logging.INFO):
    logger.setLevel(level)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(output, "check_dataflow.log"))
    fh.setLevel(level)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s| %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)


def create_dataflow_checker():

    def _update(engine, batch):
        return batch

    return Engine(_update)


def save_conf(config_file, logger, writer):
    conf_str = """
        Check dataflow configuration file:

    """
    with open(config_file, 'r') as reader:
        lines = reader.readlines()
        for l in lines:
            conf_str += l
    conf_str += "\n\n"
    logger.info(conf_str)
    writer.add_text('Configuration', conf_str)


def load_config(config_filepath):
    assert os.path.exists(config_filepath), "Configuration file '{}' is not found".format(config_filepath)
    # Handle local modules
    sys.path.insert(0, os.path.dirname(__file__))
    # Load custom module
    spec = util.spec_from_file_location("config", config_filepath)
    custom_module = util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    config = custom_module.__dict__
    assert "DATA_LOADER" in config, "DATA_LOADER parameter is not found in configuration file"
    assert "OUTPUT_PATH" in config, "OUTPUT_PATH is not found in the configuration file"
    assert "N_EPOCHS" in config, "Number of epochs should be specified in the configuration file"
    return config


def create_fig_target_distribution_per_batch(y_counts_df, n_classes, n_classes_per_fig=20):
    n = int(np.ceil(n_classes / n_classes_per_fig))
    m = min(3, n)
    k = int(np.ceil(n / m))

    fig, axarr = plt.subplots(k, m, figsize=(20, 20))

    for c in range(min(k * m, n)):
        i, j = np.unravel_index(c, dims=(k, m))
        classes = y_counts_df.columns[c * n_classes_per_fig:(c + 1) * n_classes_per_fig]
        axarr[i, j].set_title('Target distribution per batch')
        axarr[i, j].set_xlabel('Count')
        sns.boxplot(data=y_counts_df[classes], orient='h', ax=axarr[i, j])

    return fig


# def create_fig_samples_min_avg_max_per_batch(x_stats_df):
#
#     fig, axarr = plt.subplots(n_channels, 3, figsize=(20, 20))
#     fig.suptitle("Sample min/avg/max per bands")
#     with sns.axes_style("whitegrid"):
#         for i in range(n_channels):
#             for j, col in enumerate([min_cols, avg_cols, max_cols]):
#                 axarr[i, j].set_title(col[i])
#                 axarr[i, j].hist(df[col[i]], bins=100)
#     return fig
#     sns.countplot(data=df, x='shape')
#     sns.countplot(data=df, x='dtype')


def run(config_file):
    print("--- Check dataflow  --- ")

    print("Load config file ... ")
    config = load_config(config_file)

    seed = config.get("SEED", 2018)
    random.seed(seed)
    torch.manual_seed(seed)

    output = config["OUTPUT_PATH"]
    debug = config.get("DEBUG", False)

    from datetime import datetime
    now = datetime.now()
    log_dir = os.path.join(output, "check_dataflow_{}".format(now.strftime("%Y%m%d_%H%M")))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
        print("Activated debug mode")

    logger = logging.getLogger("Check dataflow")
    setup_logger(logger, log_dir, log_level)

    logger.debug("Setup tensorboard writer")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    save_conf(config_file, logger, writer)

    cuda = torch.cuda.is_available()
    if cuda:
        logger.debug("CUDA is enabled")
        from torch.backends import cudnn
        cudnn.benchmark = True

    logger.debug("Setup data loader")
    data_loader = config["DATA_LOADER"]

    logger.debug("Setup ignite dataflow checker")
    dataflow_checker = create_dataflow_checker()

    logger.debug("Setup handlers")
    # Setup timer to measure training time
    timer = Timer(average=True)
    timer.attach(dataflow_checker,
                 start=Events.EPOCH_STARTED,
                 pause=Events.ITERATION_COMPLETED,
                 resume=Events.ITERATION_STARTED)

    n_classes = 200
    n_batches = len(data_loader)

    n_channels = 3
    y_counts_per_batch = np.zeros((n_batches, n_classes), dtype=np.int)
    x_mins_per_batch = np.zeros((n_batches, n_channels), dtype=np.float)
    x_maxs_per_batch = np.zeros((n_batches, n_channels), dtype=np.float)
    x_avgs_per_batch = np.zeros((n_batches, n_channels), dtype=np.float)
    x_shapes_per_batch = np.empty((n_batches, 1), dtype=np.object)
    x_dtypes_per_batch = np.empty((n_batches, 1), dtype=np.object)

    def log_dataflow_iteration(engine, y_counts_per_batch):
        x, y = engine.state.output
        curr_iter = engine.state.iteration - 1
        y_counts_per_batch[curr_iter, :] = np.bincount(y.numpy(), minlength=n_classes)
        for i in range(n_channels):
            x_mins_per_batch[curr_iter, i] = x[:, i, :, :].min()
            x_maxs_per_batch[curr_iter, i] = x[:, i, :, :].max()
            x_avgs_per_batch[curr_iter, i] = torch.mean(x[:, i, :, :])
        x_shapes_per_batch[curr_iter, 0] = str(list(x.shape))
        x_dtypes_per_batch[curr_iter, 0] = type(x).__name__
        # if curr_iter % 100 == 0:
        #     logger.debug("Mins={} | Maxs={}".format(
        #         [x[:, i, :, :].min() for i in range(n_channels)],
        #         [x[:, i, :, :].max() for i in range(n_channels)]))

    dataflow_checker.add_event_handler(Events.ITERATION_COMPLETED, log_dataflow_iteration, y_counts_per_batch)

    def log_dataflow_epoch(engine):
        logger.info("One epoch dataflow time (seconds): {}".format(timer.value()))

    dataflow_checker.add_event_handler(Events.EPOCH_COMPLETED, log_dataflow_epoch)

    n_epochs = config["N_EPOCHS"]
    logger.debug("Start dataflow checking: {} epochs".format(n_epochs))
    try:
        dataflow_checker.run(data_loader, max_epochs=n_epochs)
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

    logger.debug("Dataflow check is ended")
    writer.close()

    cols = ["class_{}".format(i) for i in range(n_classes)]
    y_counts_df = pd.DataFrame(y_counts_per_batch, columns=cols)
    y_counts_df.to_csv(os.path.join(log_dir, "y_counts_per_batch.csv"), index=False)

    # Save figure with total target distribution
    fig = create_fig_target_distribution_per_batch(y_counts_df=y_counts_df, n_classes=n_classes, n_classes_per_fig=20)
    fig.savefig(os.path.join(log_dir, "target_distribution_per_batch.png"))
    y_counts_df = None
    y_counts_per_batch = None

    min_cols = ["b{}_min".format(i) for i in range(n_channels)]
    avg_cols = ["b{}_avg".format(i) for i in range(n_channels)]
    max_cols = ["b{}_max".format(i) for i in range(n_channels)]
    cols = min_cols + avg_cols +  max_cols + ["shape", "dtype"]
    x_stats_df = pd.DataFrame(columns=cols, index=np.arange(n_batches))
    x_stats_df.loc[:, min_cols] = x_mins_per_batch
    x_stats_df.loc[:, avg_cols] = x_avgs_per_batch
    x_stats_df.loc[:, max_cols] = x_maxs_per_batch
    x_stats_df.loc[:, "shape"] = x_shapes_per_batch
    x_stats_df.loc[:, "dtype"] = x_dtypes_per_batch
    x_stats_df.to_csv(os.path.join(log_dir, "x_stats_df.csv"), index=False)

    # Save figure with total target distribution
    # fig = create_fig_samples_min_avg_max_per_batch(x_stats_df=x_stats_df)
    # fig.savefig(os.path.join(log_dir, "samples_min_avg_max_per_batch.png"))
    # y_counts_df = None
    # y_counts_per_batch = None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_file", type=str, help="Configuration file. See examples in configs/")
    args = parser.parse_args()
    run(args.config_file)

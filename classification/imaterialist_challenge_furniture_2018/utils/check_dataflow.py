
from pathlib import Path
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

from tqdm import tqdm

import torch

from ignite.engines import Events, Engine
from ignite.handlers import Timer

# Load common module
sys.path.insert(0, Path(__file__).absolute().parent.parent.as_posix())
from common import setup_logger, save_conf


def create_dataflow_checker():

    def _update(engine, batch):
        return batch

    return Engine(_update)


def load_config(config_filepath):
    assert Path(config_filepath).exists(), "Configuration file '{}' is not found".format(config_filepath)
    # Load custom module
    spec = util.spec_from_file_location("config", config_filepath)
    custom_module = util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    config = custom_module.__dict__
    assert "DATA_LOADER" in config, "DATA_LOADER parameter is not found in configuration file"
    assert "OUTPUT_PATH" in config, "OUTPUT_PATH is not found in the configuration file"
    assert "N_EPOCHS" in config, "Number of epochs N_EPOCHS should be specified in the configuration file"
    assert "N_CLASSES" in config, "Number of classes N_CLASSES should be specified in the configuration file"
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


def create_fig_samples_min_avg_max_per_batch(x_stats_df, min_cols, avg_cols, max_cols):
    n_channels = len(min_cols)
    fig, axarr = plt.subplots(n_channels, 3, figsize=(20, 20))
    fig.suptitle("Sample min/avg/max per bands")

    with sns.axes_style("whitegrid"):
        for i in range(n_channels):
            for j, col in enumerate([min_cols, avg_cols, max_cols]):
                axarr[i, j].set_title(col[i])
                axarr[i, j].hist(x_stats_df[col[i]], bins=100)
    return fig


def run(config_file):
    print("--- Check dataflow  --- ")

    print("Load config file ... ")
    config = load_config(config_file)

    seed = config.get("SEED", 2018)
    random.seed(seed)
    torch.manual_seed(seed)

    output = Path(config["OUTPUT_PATH"])
    debug = config.get("DEBUG", False)

    from datetime import datetime
    now = datetime.now()
    log_dir = output / ("check_dataflow_{}".format(now.strftime("%Y%m%d_%H%M")))
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
        print("Activated debug mode")

    logger = logging.getLogger("Check dataflow")
    setup_logger(logger, (log_dir / "check.log").as_posix(), log_level)

    save_conf(config_file, log_dir, logger)

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

    n_classes = config["N_CLASSES"]
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
        x_shapes_per_batch[curr_iter, 0] = str(list(x.shape[1:]))
        x_dtypes_per_batch[curr_iter, 0] = type(x).__name__

        if curr_iter % 100 == 0:
            logger.debug("Iteration[{}/{}]".format(curr_iter, len(data_loader)))

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
        exit(0)
    except Exception as e:  # noqa
        logger.exception("")
        if debug:
            try:
                # open an ipython shell if possible
                import IPython
                IPython.embed()  # noqa
            except ImportError:
                print("Failed to start IPython console")
        raise e

    logger.debug("Dataflow check is ended")

    logger.debug("Create and write y_counts_per_batch.csv")
    cols = ["class_{}".format(i) for i in range(n_classes)]
    y_counts_df = pd.DataFrame(y_counts_per_batch, columns=cols)
    y_counts_df.to_csv((log_dir / "y_counts_per_batch.csv").as_posix(), index=False)

    # Save figure of total target distributions
    logger.debug("Save figure of total target distributions")
    fig = create_fig_target_distribution_per_batch(y_counts_df=y_counts_df, n_classes=n_classes, n_classes_per_fig=20)
    fig.savefig((log_dir / "target_distribution_per_batch.png").as_posix())
    y_counts_df = None
    y_counts_per_batch = None

    logger.debug("Create and write x_stats_df.csv")
    min_cols = ["b{}_min".format(i) for i in range(n_channels)]
    avg_cols = ["b{}_avg".format(i) for i in range(n_channels)]
    max_cols = ["b{}_max".format(i) for i in range(n_channels)]
    cols = min_cols + avg_cols + max_cols + ["shape", "dtype"]
    x_stats_df = pd.DataFrame(columns=cols, index=np.arange(n_batches), dtype=np.float)
    x_stats_df[min_cols] = x_mins_per_batch
    x_stats_df[avg_cols] = x_avgs_per_batch
    x_stats_df[max_cols] = x_maxs_per_batch
    x_stats_df["shape"] = x_shapes_per_batch
    x_stats_df["dtype"] = x_dtypes_per_batch
    x_stats_df.to_csv((log_dir / "x_stats_df.csv").as_posix(), index=False)

    # Save figure with sample mins, avgs, maxs
    logger.debug("Save figure with sample mins, avgs, maxs")
    fig = create_fig_samples_min_avg_max_per_batch(x_stats_df, min_cols, avg_cols, max_cols)
    fig.savefig((log_dir / "samples_min_avg_max_per_batch.png").as_posix())

    logger.debug("Save figure with sample shapes")
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    sns.countplot(data=x_stats_df, x="shape", ax=ax)
    fig.savefig((log_dir / "samples_shape_per_batch.png").as_posix())

    logger.debug("Save figure with sample dtypes")
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)
    sns.countplot(data=x_stats_df, x="dtype", ax=ax)
    fig.savefig((log_dir / "samples_dtype_per_batch.png").as_posix())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_file", type=str, help="Configuration file. See examples in configs/")
    args = parser.parse_args()
    run(args.config_file)

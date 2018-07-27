from pathlib import Path
import sys
from argparse import ArgumentParser
import random
import logging
from importlib import util
import shutil

import numpy as np
import pandas as pd

import torch
from torch.nn import functional as F
from torch.nn import Module

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engine import Events, Engine
from ignite.handlers import Timer
from ignite._utils import convert_tensor

# Load common module
sys.path.insert(0, Path(__file__).absolute().parent.parent.as_posix())
from common import setup_logger, save_conf


def create_inferencer(model, device='cpu'):

    def _prepare_batch(batch):
        x, index = batch
        x = convert_tensor(x, device=device)
        return x, index

    def _update(engine, batch):
        x, indices = _prepare_batch(batch)
        y_pred = model(x)
        y_pred = F.softmax(y_pred, dim=1)
        return {
            "y_pred": convert_tensor(y_pred, device='cpu'),
            "indices": indices
        }

    model.eval()
    inferencer = Engine(_update)
    return inferencer


def load_config(config_filepath):
    assert Path(config_filepath).exists(), "Configuration file '{}' is not found".format(config_filepath)
    # Load custom module
    spec = util.spec_from_file_location("config", config_filepath)
    custom_module = util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    config = custom_module.__dict__
    assert "TEST_LOADER" in config, "TEST_LOADER parameter is not found in configuration file"
    assert "OUTPUT_PATH" in config, "OUTPUT_PATH is not found in the configuration file"
    assert "N_CLASSES" in config, "N_CLASSES is not found in the configuration file"

    if "SAVE_PROBAS" not in config:
        config["SAVE_PROBAS"] = False
    if not config["SAVE_PROBAS"]:
        assert "SAMPLE_SUBMISSION_PATH" in config, "SAMPLE_SUBMISSION_PATH is not found in the configuration file"
        assert Path(config["SAMPLE_SUBMISSION_PATH"]).exists(), \
            "File '{}' is not found".format(config["SAMPLE_SUBMISSION_PATH"])

    if "N_TTA" not in config:
        config["N_TTA"] = 5

    assert "MODEL" in config, "MODEL is not found in configuration file"
    assert isinstance(config["MODEL"], str) and Path(config["MODEL"]).is_file(), \
        "Model is not found at '{}'".format(config["MODEL"])
    config["MODEL"] = torch.load(config["MODEL"])
    assert isinstance(config["MODEL"], Module), \
        "Model should be an instance of torch.nn.Module, but given {}".format(type(config["MODEL"]))

    # Disable requires grad:
    for param in config["MODEL"].parameters():
        param.requires_grad = False

    return config


def write_submission(indices, y_preds, sample_submission_path, submission_filepath):
    df = pd.read_csv(sample_submission_path, index_col='id')
    df.loc[indices, 'predicted'] = y_preds
    df.to_csv(submission_filepath)


def write_probas(indices, y_probas, output_filepath):
    n_samples, n_classes = y_probas.shape
    cols = ["c{}".format(i) for i in range(n_classes)]
    df = pd.DataFrame(columns=cols, index=indices)
    df.loc[:, :] = y_probas
    df.to_csv(output_filepath, index_label="id")


def run(config_file):

    print("--- iMaterialist 2018 : Inference --- ")

    print("Load config file ... ")
    config = load_config(config_file)

    seed = config.get("SEED", 2018)
    random.seed(seed)
    torch.manual_seed(seed)

    output = Path(config["OUTPUT_PATH"])
    debug = config.get("DEBUG", False)

    from datetime import datetime
    now = datetime.now()
    # log_dir = output / "inference_{}_{}".format(model_name, now.strftime("%Y%m%d_%H%M"))
    log_dir = output / ("{}".format(Path(config_file).stem)) / "{}".format(now.strftime("%Y%m%d_%H%M"))
    assert not log_dir.exists(), \
        "Output logging directory '{}' already existing".format(log_dir)
    log_dir.mkdir(parents=True)

    shutil.copyfile(config_file, (log_dir / Path(config_file).name).as_posix())

    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
        print("Activated debug mode")

    logger = logging.getLogger("iMaterialist 2018: Inference")
    setup_logger(logger, (log_dir / "predict.log").as_posix(), log_level)

    logger.debug("Setup tensorboard writer")
    writer = SummaryWriter(log_dir=(log_dir / "tensorboard").as_posix())

    save_conf(config_file, log_dir.as_posix(), logger, writer)

    model = config["MODEL"]
    device = config.get("DEVICE", 'cuda')
    if 'cuda' in device:
        assert torch.cuda.is_available(), \
            "Device {} is not compatible with torch.cuda.is_available()".format(device)
        from torch.backends import cudnn
        cudnn.benchmark = True
        logger.debug("CUDA is enabled")
        model = model.to(device)

    logger.debug("Setup test dataloader")
    test_loader = config["TEST_LOADER"]

    logger.debug("Setup ignite inferencer")
    inferencer = create_inferencer(model, device=device)

    n_tta = config["N_TTA"]
    n_classes = config["N_CLASSES"]
    batch_size = test_loader.batch_size

    logger.debug("Setup handlers")
    # Setup timer to measure evaluation time
    timer = Timer(average=True)
    timer.attach(inferencer,
                 start=Events.STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED)

    n_samples = len(test_loader.dataset)
    indices = np.zeros((n_samples, ), dtype=np.int)
    y_probas_tta = np.zeros((n_samples, n_classes, n_tta))

    @inferencer.on(Events.EPOCH_COMPLETED)
    def log_tta(engine):
        logger.debug("TTA {} / {}".format(engine.state.epoch, n_tta))

    @inferencer.on(Events.ITERATION_COMPLETED)
    def save_results(engine):
        output = engine.state.output
        tta_index = engine.state.epoch - 1
        start_index = ((engine.state.iteration - 1) % len(test_loader)) * batch_size
        end_index = min(start_index + batch_size, n_samples)
        batch_y_probas = output['y_pred'].detach().numpy()
        y_probas_tta[start_index:end_index, :, tta_index] = batch_y_probas
        if tta_index == 0:
            indices[start_index:end_index] = output['indices']

    logger.info("Start inference")
    try:
        inferencer.run(test_loader, max_epochs=n_tta)
    except KeyboardInterrupt:
        logger.info("Catched KeyboardInterrupt -> exit")
        return
    except Exception as e:  # noqa
        logger.exception("")
        if debug:
            try:
                # open an ipython shell if possible
                import IPython
                IPython.embed()  # noqa
            except ImportError:
                print("Failed to start IPython console")
        return

    # Average probabilities:
    y_probas = np.mean(y_probas_tta, axis=-1)

    if config["SAVE_PROBAS"]:
        logger.info("Write probabilities file")
        probas_filepath = log_dir / "probas.csv"
        write_probas(indices, y_probas, probas_filepath)
    else:
        y_preds = np.argmax(y_probas, axis=-1) + 1  # as labels are one-based
        logger.info("Write submission file")
        submission_filepath = log_dir / "predictions.csv"
        sample_submission_path = config["SAMPLE_SUBMISSION_PATH"]
        write_submission(indices, y_preds, sample_submission_path, submission_filepath)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_file", type=str, help="Configuration file. See examples in configs/")
    args = parser.parse_args()
    run(args.config_file)

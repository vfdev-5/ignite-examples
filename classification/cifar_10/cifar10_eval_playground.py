from __future__ import print_function

import os
from argparse import ArgumentParser
import random
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip
from torchvision.datasets import CIFAR10

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engines import Events, create_supervised_evaluator, Engine
from ignite.metrics import CategoricalAccuracy, Loss, Precision, Recall
from ignite.handlers import ModelCheckpoint, Timer, EarlyStopping
from ignite._utils import to_variable, to_tensor

from models import get_small_squeezenet_v1_1


SEED = 12345
random.seed(SEED)
torch.manual_seed(SEED)


def get_test_data_loader(path, batch_size, num_workers, cuda=True):

    test_data_transform = Compose([
        Resize(42),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_dataset = CIFAR10(path, train=False, transform=test_data_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=cuda)
    return test_loader


def create_logger(output, level=logging.INFO):
    logger = logging.getLogger("Cifar10 Playground: Eval")
    logger.setLevel(level)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(output, "eval.log"))
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
    return logger


def create_summary_writer(model, log_dir, cuda):
    writer = SummaryWriter(log_dir=log_dir)
    try:
        dummy_input = to_variable(torch.rand(10, 3, 42, 42), cuda=cuda)
        writer.add_graph(model, dummy_input)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def create_inferencer(model, cuda=False):

    def _prepare_batch(batch):
        x, _ = batch
        x = to_variable(x, cuda=cuda)
        return x, _

    def _update(engine, batch):
        model.eval()
        x, _ = _prepare_batch(batch)
        y_pred = model(x)
        return y_pred

    inferencer = Engine(_update)
    return inferencer


def load_checkpoint(filename, model):
    state = torch.load(filename)
    model.load_state_dict(state['state_dict'])


def run(checkpoint_path, path, batch_size, num_workers, output, debug):

    print("--- Cifar10 Playground : Eval --- ")

    from datetime import datetime
    now = datetime.now()
    log_dir = os.path.join(output, "%s" % (now.strftime("%Y%m%d_%H%M")))
    os.makedirs(log_dir, exist_ok=True)

    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
        print("Activated debug mode")
    logger = create_logger(log_dir, log_level)

    cuda = torch.cuda.is_available()
    if cuda:
        from torch.backends import cudnn
        cudnn.benchmark = True

    logger.debug("Setup test dataloader")
    test_loader = get_test_data_loader(path, batch_size, num_workers, cuda=cuda)

    logger.debug("Setup model")
    model = get_small_squeezenet_v1_1(num_classes=10)
    model_name = model.__class__.__name__
    if cuda:
        model = model.cuda()

    load_checkpoint(checkpoint_path, model)

    logger.debug("Setup tensorboard writer")
    writer = create_summary_writer(model, os.path.join(log_dir, "tensorboard"), cuda=cuda)

    logger.debug("Setup ignite trainer and evaluator")
    inferencer = create_inferencer(model, cuda=cuda)

    logger.debug("Setup handlers")
    # Setup timer to measure evaluation time
    timer = Timer(average=True)
    timer.attach(inferencer,
                 start=Events.STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED)

    logger.debug("Start evaluation")
    try:
        inferencer.run(test_loader, max_epochs=1)
    except KeyboardInterrupt:
        logger.info("Catched KeyboardInterrupt -> exit")
    except Exception as e:  # noqa
        logger.exception("")
        if args.debug:
            try:
                # open an ipython shell if possible
                import IPython
                IPython.embed()  # noqa
            except ImportError:
                print("Failed to start IPython console")

    writer.close()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Model checkpoint to load")
    parser.add_argument("--path", type=str, default=".",
                        help="Optional path to Cifar10 dataset")
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers in data loader(default: 8)')
    parser.add_argument("--output", type=str, default="output",
                        help="directory to store best models")
    parser.add_argument("--debug", action="store_true", default=0,
                        help="Enable debugging")

    args = parser.parse_args()
    run(args.checkpoint, args.path, args.batch_size, args.num_workers, args.output, args.debug)

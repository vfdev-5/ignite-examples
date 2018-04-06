from __future__ import print_function

import os
from argparse import ArgumentParser
import random
import logging

import numpy as np

from sklearn.metrics import classification_report

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip
from torchvision.datasets import CIFAR10

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engines import Events, Engine
from ignite.handlers import Timer
from ignite._utils import to_variable, to_tensor


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

    class IndexedDataset(Dataset):

        def __init__(self, ds):
            super(IndexedDataset, self).__init__()
            self.ds = ds

        def __getitem__(self, index):
            return self.ds[index], index

        def __len__(self):
            return len(self.ds)

    test_dataset = IndexedDataset(CIFAR10(path, train=False, transform=test_data_transform, download=True))
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False,
                             num_workers=num_workers, pin_memory=cuda)
    return test_loader, test_dataset


def create_logger(output, level=logging.INFO):
    logger = logging.getLogger("Cifar10 Playground: Inference")
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


def create_inferencer(model, cuda=True):

    def _prepare_batch(batch):
        (x, y), indices = batch
        x = to_variable(x, cuda=cuda)
        return x, y, indices

    def _update(engine, batch):
        x, y, indices = _prepare_batch(batch)
        y_pred = model(x)
        return {
            "x": to_tensor(x, cpu=True),
            "y_pred": to_tensor(y_pred, cpu=True),
            "y_true": y,
            "indices": indices
        }

    model.eval()
    inferencer = Engine(_update)
    return inferencer


def run(checkpoint, dataset_path, batch_size, num_workers, n_tta, output, debug):

    print("--- Cifar10 Playground : Inference --- ")

    from datetime import datetime
    now = datetime.now()
    log_dir = os.path.join(output, "inference_%s" % (now.strftime("%Y%m%d_%H%M")))
    os.makedirs(log_dir, exist_ok=True)

    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
        print("Activated debug mode")
    logger = create_logger(log_dir, log_level)

    cuda = torch.cuda.is_available()
    if cuda:
        logger.debug("CUDA is enabled")
        from torch.backends import cudnn
        cudnn.benchmark = True

    logger.debug("Setup test dataloader")
    test_loader, test_dataset = get_test_data_loader(dataset_path, batch_size, num_workers, cuda=cuda)

    logger.debug("Setup model: {}".format(checkpoint))
    model = torch.load(checkpoint)
    # model_name = model.__class__.__name__
    if cuda:
        model = model.cuda()

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

    indices = np.zeros((len(test_dataset), n_tta), dtype=np.int32)
    y_probas_tta = np.zeros((len(test_dataset), 10, n_tta))
    y_true = np.zeros((len(test_dataset), ), dtype=np.int32)

    @inferencer.on(Events.EPOCH_COMPLETED)
    def log_tta(engine):
        logger.debug("TTA {} / {}".format(engine.state.epoch, n_tta))

    @inferencer.on(Events.ITERATION_COMPLETED)
    def save_results(engine):
        output = engine.state.output
        tta_index = engine.state.epoch - 1
        start_index = ((engine.state.iteration - 1) % len(test_loader)) * batch_size
        batch_indices = output['indices'].numpy()
        batch_y_probas = output['y_pred'].numpy()
        batch_y_true = output['y_true'].numpy()
        end_index = min(start_index + batch_size, len(indices))
        indices[start_index:end_index, tta_index] = batch_indices
        y_probas_tta[start_index:end_index, :, tta_index] = batch_y_probas
        if tta_index == 0:
            y_true[start_index:end_index] = batch_y_true

    logger.debug("Start inference")
    try:
        inferencer.run(test_loader, max_epochs=n_tta)
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

    # Check indices:
    for i in range(n_tta - 1):
        ind1 = indices[:, i]
        ind2 = indices[:, i + 1]
        assert (ind1 == ind2).all()

    # Average probabilities:
    y_probas = np.mean(y_probas_tta, axis=-1)
    y_preds = np.argmax(y_probas, axis=-1)

    logger.info("\n" + classification_report(y_true, y_preds))

    writer.close()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Model checkpoint to load")
    parser.add_argument("--path", type=str, default=".",
                        help="Optional path to Cifar10 dataset")
    parser.add_argument('--n_tta', type=int, default=5,
                        help='Number of test time augmentations (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers in data loader(default: 8)')
    parser.add_argument("--output", type=str, default="output",
                        help="directory to store best models")
    parser.add_argument("--debug", action="store_true", default=0,
                        help="Enable debugging")

    args = parser.parse_args()
    run(args.checkpoint, args.path,
        args.batch_size, args.num_workers,
        args.n_tta,
        args.output, args.debug)
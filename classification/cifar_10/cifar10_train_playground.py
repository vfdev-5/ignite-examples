from __future__ import print_function

import os
from argparse import ArgumentParser
import random
import logging

import numpy as np

from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, RandomChoice, RandomAffine
from torchvision.transforms import ColorJitter
from torchvision.datasets import CIFAR10

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engines import Events, create_supervised_evaluator, Engine
from ignite.metrics import CategoricalAccuracy, Loss, Precision, Recall
from ignite.handlers import ModelCheckpoint, Timer, EarlyStopping
from ignite._utils import to_variable, to_tensor

from models import get_small_squeezenet_v1_1, get_small_squeezenet_v1_1_bn
from lr_schedulers import LRSchedulerWithRestart


SEED = 12345
random.seed(SEED)
torch.manual_seed(SEED)

MODEL_MAP = {
    "squeezenet_v1_1": get_small_squeezenet_v1_1,
    "squeezenet_v1_1_bn": get_small_squeezenet_v1_1_bn
}


def get_train_val_indices(dataset, fold_index=0, n_splits=5):
    # Stratified split: train/val:
    n_samples = len(dataset)
    X = np.zeros((n_samples, 1))
    y = np.zeros(n_samples)
    for i, (_, label) in enumerate(dataset):
        y[i] = label

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for i, (train_indices, val_indices) in enumerate(skf.split(X, y)):
        if i == fold_index:
            return train_indices, val_indices


def get_data_loaders(path, train_batch_size, val_batch_size, num_workers, cuda=True):
    train_data_transform = Compose([
        Resize(42),
        RandomChoice([
            RandomAffine(degrees=(-50, 50), scale=(0.95, 1.05), translate=(0.05, 0.05)),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ]),
        ColorJitter(hue=0.05),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    val_data_transform = Compose([
        Resize(42),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = CIFAR10(path, train=True, transform=train_data_transform, download=True)
    val_dataset = CIFAR10(path, train=True, transform=val_data_transform, download=False)

    train_indices, val_indices = get_train_val_indices(train_dataset, fold_index=0, n_splits=5)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              sampler=SubsetRandomSampler(train_indices),
                              num_workers=num_workers, pin_memory=cuda)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size,
                            sampler=SubsetRandomSampler(val_indices),
                            num_workers=num_workers, pin_memory=cuda)

    return train_loader, val_loader


def create_logger(output, level=logging.INFO):
    logger = logging.getLogger("Cifar10 Playground: Train")
    logger.setLevel(level)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(output, "train.log"))
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


def create_supervised_trainer(model, optimizer, loss_fn, metrics={}, cuda=False):

    def _prepare_batch(batch):
        x, y = batch
        x = to_variable(x, cuda=cuda)
        y = to_variable(y, cuda=cuda)
        return x, y

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return {
            'loss': loss.data.cpu()[0],
            'y_pred': y_pred,
            'y': y
        }

    trainer = Engine(_update)

    for name, metric in metrics.items():
        trainer.add_event_handler(Events.EPOCH_STARTED, metric.started)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, metric.iteration_completed)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, metric.completed, name)

    return trainer


def run(path, model_name,
        train_batch_size, val_batch_size, num_workers,
        epochs, lr, gamma, restart_every, restart_factor, init_lr_factor,
        log_interval, output, debug):

    print("--- Cifar10 Playground : Train --- ")

    from datetime import datetime
    now = datetime.now()
    log_dir = os.path.join(output, "training_%s" % (now.strftime("%Y%m%d_%H%M")))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
        print("Activated debug mode")
    logger = create_logger(log_dir, log_level)

    cuda = torch.cuda.is_available()
    if cuda:
        from torch.backends import cudnn
        cudnn.benchmark = True

    logger.debug("Setup model: {}".format(model_name))

    if not os.path.isfile(model_name):
        assert model_name in MODEL_MAP, "Model name not in {}".format(MODEL_MAP.keys())
        model = MODEL_MAP[model_name](num_classes=10)
    else:
        model = torch.load(model_name)

    model_name = model.__class__.__name__
    if cuda:
        model = model.cuda()

    logger.debug("Setup train/val dataloaders")
    train_loader, val_loader = get_data_loaders(path, train_batch_size, val_batch_size, num_workers, cuda=cuda)

    logger.debug("Setup tensorboard writer")
    writer = create_summary_writer(model, os.path.join(log_dir, "tensorboard"), cuda=cuda)

    logger.debug("Setup optimizer")
    optimizer = Adam(model.parameters(), lr=lr)

    logger.debug("Setup criterion")
    criterion = nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()

    lr_scheduler = ExponentialLR(optimizer, gamma=gamma)
    lr_scheduler_restarts = LRSchedulerWithRestart(lr_scheduler,
                                                   restart_every=restart_every,
                                                   restart_factor=restart_factor,
                                                   init_lr_factor=init_lr_factor)
    reduce_on_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

    def output_transform(output):
        y_pred = output['y_pred']
        y = output['y']
        return to_tensor(y_pred, cpu=not cuda), to_tensor(y, cpu=not cuda)

    logger.debug("Setup ignite trainer and evaluator")
    trainer = create_supervised_trainer(model, optimizer, criterion,
                                        metrics={
                                            'accuracy': CategoricalAccuracy(output_transform=output_transform),
                                            'precision': Precision(output_transform=output_transform),
                                            'recall': Recall(output_transform=output_transform),
                                        },
                                        cuda=cuda)
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'accuracy': CategoricalAccuracy(),
                                                'precision': Precision(),
                                                'recall': Recall(),
                                                'nll': Loss(criterion)
                                            },
                                            cuda=cuda)

    logger.debug("Setup handlers")
    # Setup timer to measure training time
    timer = Timer(average=True)
    timer.attach(trainer,
                 start=Events.EPOCH_STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(engine.state.epoch, iter,
                                                                         len(train_loader),
                                                                         engine.state.output['loss']))

            writer.add_scalar("training/loss", engine.state.output['loss'], engine.state.iteration)

    @trainer.on(Events.EPOCH_STARTED)
    def update_lr_schedulers(engine):
        lr_scheduler_restarts.step()
        lrs = lr_scheduler_restarts.get_lr()
        if len(lrs) == 1:
            writer.add_scalar("learning_rate", lrs[0], engine.state.epoch)
        else:
            for i, lr in enumerate(lrs):
                writer.add_scalar("learning_rate_group_{}".format(i), lr, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_metrics(engine):

        logger.info("One epoch training time (seconds): {}".format(timer.value()))
        logger.info("Training Results - Epoch: {}  Avg accuracy: {:.2f}"
                    .format(engine.state.epoch, engine.state.metrics['accuracy']))
        writer.add_scalar("training/accuracy", engine.state.metrics['accuracy'], engine.state.epoch)

        for metric_name in ['precision', 'recall']:
            value = engine.state.metrics[metric_name].cpu() if cuda else engine.state.metrics[metric_name]
            avg_value = torch.mean(engine.state.metrics[metric_name])
            writer.add_scalar("training/avg_{}".format(metric_name), avg_value, engine.state.epoch)
            logger.info("   {} per class: {}".format(metric_name, value.numpy().tolist()))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        metrics = evaluator.run(val_loader).metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        writer.add_scalar("validation/loss", avg_nll, engine.state.epoch)
        writer.add_scalar("validation/accuracy", avg_accuracy, engine.state.epoch)
        logger.info("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                    .format(engine.state.epoch, avg_accuracy, avg_nll))

        for metric_name in ['precision', 'recall']:
            value = engine.state.metrics[metric_name].cpu() if cuda else engine.state.metrics[metric_name]
            avg_value = torch.mean(engine.state.metrics[metric_name])
            writer.add_scalar("validation/avg_{}".format(metric_name), avg_value, engine.state.epoch)
            logger.info("   {} per class: {}".format(metric_name, value.numpy().tolist()))

    @evaluator.on(Events.COMPLETED)
    def update_reduce_on_plateau(engine):
        val_loss = engine.state.metrics['nll']
        reduce_on_plateau.step(val_loss)

    @evaluator.on(Events.COMPLETED)
    def update_early_stopping(engine):
        val_loss = engine.state.metrics['nll']
        reduce_on_plateau.step(val_loss)

    def score_function(engine):
        val_loss = engine.state.metrics['nll']
        # Objects with highest scores will be retained.
        return -val_loss

    # Setup early stopping:
    handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    # Setup model checkpoint:
    handler = ModelCheckpoint(log_dir,
                              filename_prefix="model",
                              score_name="val_loss",
                              score_function=score_function,
                              n_saved=5,
                              atomic=True,
                              create_dir=True,
                              exist_ok=True)
    evaluator.add_event_handler(Events.COMPLETED, handler, {model_name: model})

    logger.debug("Start training")
    try:
        trainer.run(train_loader, max_epochs=epochs)
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
    parser.add_argument("--path", type=str, default=".",
                        help="Optional path to Cifar10 dataset (default: .)")
    parser.add_argument("--model", type=str,
                        default="squeezenet_v1_1",
                        help="Model choice: squeezenet_v1_1, squeezenet_v1_1_bn or "
                             "path to saved model (default: squeezenet_v1_1)")
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers in data loader(default: 8)')
    parser.add_argument('--val_batch_size', type=int, default=100,
                        help='input batch size for validation (default: 100)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='learning rate exponential scheduler gamma (default: 0.95)')
    parser.add_argument('--restart_every', type=float, default=10,
                        help='restart lr scheduler every `restart_every` epoch(default: 10)')
    parser.add_argument('--restart_factor', type=float, default=1.5,
                        help='factor to rescale `restart_every` after each restart (default: 1.5)')
    parser.add_argument('--init_lr_factor', type=float, default=0.5,
                        help='factor to rescale base lr after each restart (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--output", type=str, default="output",
                        help="directory to store best models")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debugging")
    args = parser.parse_args()

    run(args.path, args.model,
        args.batch_size, args.val_batch_size, args.num_workers,
        args.epochs,
        args.lr, args.gamma, args.restart_every, args.restart_factor, args.init_lr_factor,
        args.log_interval, args.output,
        args.debug)

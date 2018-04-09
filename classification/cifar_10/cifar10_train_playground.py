from __future__ import print_function

import os
from argparse import ArgumentParser
import random
import logging
from importlib import util

import numpy as np

from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import CIFAR10

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engines import Events, create_supervised_evaluator, Engine
from ignite.metrics import CategoricalAccuracy, Loss, Precision, Recall
from ignite.handlers import ModelCheckpoint, Timer, EarlyStopping
from ignite._utils import to_variable, to_tensor

from models.small_squeezenets_v1_1 import get_small_squeezenet_v1_1, SqueezeNetV11BN
from models.small_vgg16_bn import get_small_vgg16_bn
from models.small_nasnet_a_mobile import SmallNASNetAMobile
from lr_schedulers import LRSchedulerWithRestart


SEED = 12345
random.seed(SEED)
torch.manual_seed(SEED)

MODEL_MAP = {
    "squeezenet_v1_1": get_small_squeezenet_v1_1,
    "squeezenet_v1_1_bn": SqueezeNetV11BN,
    "vgg16_bn": get_small_vgg16_bn,
    "nasnet_a_mobile": SmallNASNetAMobile
}

OPTIMIZER_MAP = {
    "adam": Adam,
    "sgd": SGD
}


def get_train_val_indices(data_loader, fold_index=0, n_splits=5):
    # Stratified split: train/val:
    n_samples = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    X = np.zeros((n_samples, 1))
    y = np.zeros(n_samples, dtype=np.int)
    for i, (_, label) in enumerate(data_loader):
        start_index = batch_size * i
        end_index = batch_size * (i + 1)
        y[start_index: end_index] = label.numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for i, (train_indices, val_indices) in enumerate(skf.split(X, y)):
        if i == fold_index:
            return train_indices, val_indices


def get_data_loaders(dataset_path, imgaugs, train_batch_size, val_batch_size, num_workers, cuda=True):

    # Load imgaugs module:
    this_dir = os.path.dirname(__file__)
    spec = util.spec_from_file_location("imgaugs", os.path.join(this_dir, imgaugs))
    custom_module = util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)

    train_data_transform = getattr(custom_module, "train_data_transform")
    val_data_transform = getattr(custom_module, "val_data_transform")

    train_data_transform = Compose(train_data_transform)
    val_data_transform = Compose(val_data_transform)

    train_dataset = CIFAR10(dataset_path, train=True, transform=train_data_transform, download=True)
    val_dataset = CIFAR10(dataset_path, train=True, transform=val_data_transform, download=False)

    # Temporary dataset and dataloader to create train/val split
    trainval_dataset = CIFAR10(dataset_path, train=True, transform=ToTensor(), download=False)
    trainval_loader = DataLoader(trainval_dataset, batch_size=train_batch_size,
                                 shuffle=False, drop_last=False,
                                 num_workers=num_workers, pin_memory=False)

    train_indices, val_indices = get_train_val_indices(trainval_loader, fold_index=0, n_splits=5)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              sampler=SubsetRandomSampler(train_indices),
                              num_workers=num_workers, pin_memory=cuda)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size,
                            sampler=SubsetRandomSampler(val_indices),
                            num_workers=num_workers, pin_memory=cuda)

    return train_loader, val_loader


def setup_logger(logger, output, level=logging.INFO):
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


def write_model_graph(writer, model, cuda):
    try:
        dummy_input = to_variable(torch.rand(10, 3, 32, 32), cuda=cuda)
        writer.add_graph(model, dummy_input)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))


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


def save_conf(logger, writer, model_name, imgaugs,
              train_batch_size, val_batch_size, num_workers,
              epochs, optim,
              lr, lr_update_every, gamma, restart_every, restart_factor, init_lr_factor,
              lr_reduce_patience, early_stop_patience,
              output):
    conf_str = """        
        Training configuration:
            Model: {model}
            Image augs: {imgaugs}
            Train batch size: {train_batch_size}
            Val batch size: {val_batch_size}
            Number of workers: {num_workers}
            Number of epochs: {epochs}
            Optimizer: {optim}
            Learning rate: {lr}
            Learning rate update every : {lr_update_every} epoch(s)
            Exp lr scheduler gamma: {gamma}
                restart every: {restart_every}
                restart factor: {restart_factor}
                init lr factor: {init_lr_factor}
            Reduce on plateau: {lr_reduce_patience}
            Early stopping patience: {early_stop_patience}
            Output folder: {output}        
    """.format(
        model=model_name,
        imgaugs=imgaugs,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        epochs=epochs,
        optim=optim,
        lr=lr,
        lr_update_every=lr_update_every,
        gamma=gamma,
        restart_every=restart_every,
        restart_factor=restart_factor,
        init_lr_factor=init_lr_factor,
        lr_reduce_patience=lr_reduce_patience,
        early_stop_patience=early_stop_patience,
        output=output
    )
    logger.info(conf_str)
    writer.add_text('Configuration', conf_str)


def run(path, model_name, imgaugs,
        train_batch_size, val_batch_size, num_workers,
        epochs, optim,
        lr, lr_update_every, gamma, restart_every, restart_factor, init_lr_factor,
        lr_reduce_patience, early_stop_patience,
        log_interval, output, debug):

    print("--- Cifar10 Playground : Training --- ")

    from datetime import datetime
    now = datetime.now()
    log_dir = os.path.join(output, "training_{}_{}".format(model_name, now.strftime("%Y%m%d_%H%M")))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
        print("Activated debug mode")

    logger = logging.getLogger("Cifar10 Playground: Train")
    setup_logger(logger, log_dir, log_level)

    logger.debug("Setup tensorboard writer")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    save_conf(logger, writer, model_name, imgaugs,
              train_batch_size, val_batch_size, num_workers,
              epochs, optim,
              lr, lr_update_every, gamma, restart_every, restart_factor, init_lr_factor,
              lr_reduce_patience, early_stop_patience,
              log_dir)

    cuda = torch.cuda.is_available()
    if cuda:
        logger.debug("CUDA is enabled")
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
    write_model_graph(writer, model=model, cuda=cuda)

    logger.debug("Setup train/val dataloaders")
    train_loader, val_loader = get_data_loaders(path, imgaugs, train_batch_size, val_batch_size, num_workers, cuda=cuda)

    logger.debug("Setup optimizer")
    assert optim in OPTIMIZER_MAP, "Optimizer name not in {}".format(OPTIMIZER_MAP.keys())
    optimizer = OPTIMIZER_MAP[optim](model.parameters(), lr=lr)

    logger.debug("Setup criterion")
    criterion = nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()

    lr_scheduler = ExponentialLR(optimizer, gamma=gamma)
    lr_scheduler_restarts = LRSchedulerWithRestart(lr_scheduler,
                                                   restart_every=restart_every,
                                                   restart_factor=restart_factor,
                                                   init_lr_factor=init_lr_factor)
    reduce_on_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                          patience=lr_reduce_patience,
                                          threshold=0.01, verbose=True)

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
                                            'nll': Loss(criterion, output_transform=output_transform)
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

            writer.add_scalar("training/loss_vs_iterations", engine.state.output['loss'], engine.state.iteration)

    @trainer.on(Events.EPOCH_STARTED)
    def update_lr_schedulers(engine):
        if (engine.state.epoch - 1) % lr_update_every == 0:
            lr_scheduler_restarts.step()

    @trainer.on(Events.EPOCH_STARTED)
    def log_lrs(engine):
        if len(optimizer.param_groups) == 1:
            lr = float(optimizer.param_groups[0]['lr'])
            writer.add_scalar("learning_rate", lr, engine.state.epoch)
            logger.debug("Learning rate: {}".format(lr))
        else:
            for i, param_group in enumerate(optimizer.param_groups):
                lr = float(param_group['lr'])
                logger.debug("Learning rate (group {}): {}".format(i, lr))
                writer.add_scalar("learning_rate_group_{}".format(i), lr, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_metrics(engine):

        logger.info("One epoch training time (seconds): {}".format(timer.value()))
        logger.info("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                    .format(engine.state.epoch, engine.state.metrics['accuracy'], engine.state.metrics['nll']))
        writer.add_scalar("training/avg_accuracy", engine.state.metrics['accuracy'], engine.state.epoch)
        writer.add_scalar("training/avg_loss", engine.state.metrics['nll'], engine.state.epoch)
        for metric_name in ['precision', 'recall']:
            value = engine.state.metrics[metric_name].cpu() if cuda else engine.state.metrics[metric_name]
            avg_value = torch.mean(value)
            writer.add_scalar("training/avg_{}".format(metric_name), avg_value, engine.state.epoch)
            logger.debug("   {} per class: {}".format(metric_name, value.numpy().tolist()))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        metrics = evaluator.run(val_loader).metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        writer.add_scalar("validation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)
        logger.info("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                    .format(engine.state.epoch, avg_accuracy, avg_nll))

        for metric_name in ['precision', 'recall']:
            value = engine.state.metrics[metric_name].cpu() if cuda else engine.state.metrics[metric_name]
            avg_value = torch.mean(engine.state.metrics[metric_name])
            writer.add_scalar("validation/avg_{}".format(metric_name), avg_value, engine.state.epoch)
            logger.debug("   {} per class: {}".format(metric_name, value.numpy().tolist()))

    @evaluator.on(Events.COMPLETED)
    def update_reduce_on_plateau(engine):
        val_loss = engine.state.metrics['nll']
        reduce_on_plateau.step(val_loss)

    def score_function(engine):
        val_loss = engine.state.metrics['nll']
        # Objects with highest scores will be retained.
        return -val_loss

    # Setup early stopping:
    handler = EarlyStopping(patience=early_stop_patience, score_function=score_function, trainer=trainer)
    setup_logger(handler._logger, log_dir, log_level)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    # Setup model checkpoint:
    best_model_saver = ModelCheckpoint(log_dir,
                                       filename_prefix="model",
                                       score_name="val_loss",
                                       score_function=score_function,
                                       n_saved=5,
                                       atomic=True,
                                       create_dir=True)
    evaluator.add_event_handler(Events.COMPLETED, best_model_saver, {model_name: model})

    last_model_saver = ModelCheckpoint(log_dir,
                                       filename_prefix="checkpoint",
                                       save_interval=1,
                                       n_saved=1,
                                       atomic=True,
                                       create_dir=True)
    evaluator.add_event_handler(Events.COMPLETED, last_model_saver, {model_name: model})

    logger.debug("Start training: {} epochs".format(epochs))
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

    logger.debug("Training is ended")
    writer.close()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=".",
                        help="Optional path to Cifar10 dataset (default: .)")
    parser.add_argument("--model", type=str,
                        default="squeezenet_v1_1",
                        help="Model choice: \n"
                             "- squeezenet_v1_1 \n"
                             "- squeezenet_v1_1_bn \n"
                             "- vgg16_bn \n "
                             "- nasnet_a_mobile \n "
                             "or a path to saved model (default: squeezenet_v1_1)")
    parser.add_argument('--imgaugs', type=str, default="imgaugs.py",
                        help='image augmentations module, python file (default: imgaugs.py)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers in data loader(default: 8)')
    parser.add_argument('--val_batch_size', type=int, default=100,
                        help='input batch size for validation (default: 100)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--optim', type=str, choices=["adam", "sgd"], default="adam",
                        help='optimizer choice: adam or sgd')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr_update_every', type=int, default=1,
                        help='learning rate update every `lr_update_every` epoch (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='learning rate exponential scheduler gamma (default: 0.95)')
    parser.add_argument('--restart_every', type=float, default=10,
                        help='restart lr scheduler every `restart_every` epoch(default: 10)')
    parser.add_argument('--restart_factor', type=float, default=1.5,
                        help='factor to rescale `restart_every` after each restart (default: 1.5)')
    parser.add_argument('--init_lr_factor', type=float, default=0.5,
                        help='factor to rescale base lr after each restart (default: 0.5)')
    parser.add_argument('--lr_reduce_patience', type=int, default=10,
                        help='reduce on plateau patience in epochs (default: 10)')
    parser.add_argument('--early_stop_patience', type=int, default=20,
                        help='early stopping patience in epochs (default: 20)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--output", type=str, default="output",
                        help="directory to store best models")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Enable debugging")
    args = parser.parse_args()

    run(args.dataset_path, args.model,
        args.imgaugs,
        args.batch_size, args.val_batch_size, args.num_workers,
        args.epochs,
        args.optim,
        args.lr, args.lr_update_every, args.gamma, args.restart_every, args.restart_factor, args.init_lr_factor,
        args.lr_reduce_patience, args.early_stop_patience,
        args.log_interval, args.output,
        args.debug)

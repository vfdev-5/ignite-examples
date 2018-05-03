from __future__ import print_function

import os
import sys
from argparse import ArgumentParser
import random
import logging
from importlib import util

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torchvision.transforms import ToTensor

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss, Precision, Recall
from ignite.handlers import ModelCheckpoint, Timer, EarlyStopping
from ignite._utils import convert_tensor

from dataflow import get_trainval_data_loaders
from common import setup_logger, save_conf
from figures import create_fig_param_per_class


def write_model_graph(writer, model, data_loader, device):
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    x = convert_tensor(x, device=device)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))


# ## Until "Trainer with metrics #165" is merged, https://github.com/pytorch/ignite/pull/165
from ignite.engine import Engine, _prepare_batch


def create_supervised_trainer(model, optimizer, loss_fn, metrics={}, device=None):
    """
    Factory function for creating a trainer for supervised models
    Args:
        model (torch.nn.Module): the model to train
        optimizer (torch.optim.Optimizer): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        metrics (dict of str: Metric): a map of metric names to Metrics
        device (optional): device type specification (default: None)
    Returns:
        Engine: a trainer engine with supervised update function
    """
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item(), y_pred, y

    def _metrics_transform(output):
        return output[1], output[2]

    engine = Engine(_update)

    for name, metric in metrics.items():
        metric._output_transform = _metrics_transform
        metric.attach(engine, name)

    return engine
# ## END OF Until "Trainer with metrics #165" is merged, https://github.com/pytorch/ignite/pull/165


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

    assert "N_EPOCHS" in config, "Number of epochs should be specified in the configuration file"

    assert "MODEL" in config, "MODEL is not found in configuration file"
    if isinstance(config["MODEL"], str) and os.path.isfile(config["MODEL"]):
        config["MODEL"] = torch.load(config["MODEL"])
    assert isinstance(config["MODEL"], nn.Module), \
        "Model should be an instance of torch.nn.Module, but given {}".format(type(config["MODEL"]))

    if "TRAIN_TRANSFORMS" not in config:
        config["TRAIN_TRANSFORMS"] = [ToTensor(), ]

    if "VAL_TRANSFORMS" not in config:
        config["VAL_TRANSFORMS"] = [ToTensor(), ]

    if "OPTIM" not in config:
        config["OPTIM"] = SGD(config["MODEL"].parameters(), lr=0.1, momentum=0.9, nesterov=True)
    assert isinstance(config["OPTIM"], torch.optim.Optimizer), \
        "Optimizer should be an instance of torch.optim.Optimizer, but given {}".format(type(config["OPTIM"]))

    if "LR_SCHEDULERS" in config:
        assert isinstance(config["LR_SCHEDULERS"], (tuple, list))
        for s in config["LR_SCHEDULERS"]:
            assert isinstance(s, _LRScheduler), \
                "LR scheduler 's' should be instance of torch.optim.lr_scheduler._LRScheduler, " \
                "but given {}".format(type(s))

    if "REDUCE_LR_ON_PLATEAU" in config:
        assert isinstance(config["REDUCE_LR_ON_PLATEAU"], ReduceLROnPlateau)

    return config


def run(config_file):

    print("--- Tiny ImageNet 200 Playground : Training --- ")

    print("Load config file ... ")
    config = load_config(config_file)

    seed = config.get("SEED", 2018)
    random.seed(seed)
    torch.manual_seed(seed)

    output = config["OUTPUT_PATH"]
    model = config["MODEL"]
    model_name = model.__class__.__name__
    debug = config.get("DEBUG", False)

    from datetime import datetime
    now = datetime.now()
    log_dir = os.path.join(output, "training_{}_{}".format(model_name, now.strftime("%Y%m%d_%H%M")))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
        print("Activated debug mode")

    logger = logging.getLogger("Tiny ImageNet 200: Train")
    setup_logger(logger, os.path.join(log_dir, "train.log"), log_level)

    logger.debug("Setup tensorboard writer")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    save_conf(config_file, log_dir, logger, writer)

    device = 'cpu'
    if torch.cuda.is_available():
        logger.debug("CUDA is enabled")
        from torch.backends import cudnn
        cudnn.benchmark = True
        device = 'cuda'
        model = model.to(device)

    logger.debug("Setup train/val dataloaders")
    dataset_path = config["DATASET_PATH"]
    train_data_transform = config["TRAIN_TRANSFORMS"]
    val_data_transform = config["VAL_TRANSFORMS"]
    train_batch_size = config.get("BATCH_SIZE", 64)
    val_batch_size = config.get("VAL_BATCH_SIZE", train_batch_size)
    num_workers = config.get("NUM_WORKERS", 8)
    trainval_split = config.get("TRAINVAL_SPLIT", {'fold_index': 0, 'n_splits': 7})
    train_loader, val_loader = get_trainval_data_loaders(dataset_path,
                                                         train_data_transform,
                                                         val_data_transform,
                                                         train_batch_size, val_batch_size,
                                                         trainval_split,
                                                         num_workers, device=device)

    write_model_graph(writer, model=model, data_loader=train_loader, device=device)

    optimizer = config["OPTIM"]

    logger.debug("Setup criterion")
    criterion = nn.CrossEntropyLoss()
    if 'cuda' in device:
        criterion = criterion.to(device)

    lr_schedulers = config.get("LR_SCHEDULERS")

    logger.debug("Setup ignite trainer and evaluator")
    trainer = create_supervised_trainer(model, optimizer, criterion,
                                        metrics={
                                            'accuracy': CategoricalAccuracy(),
                                            'nll': Loss(criterion),
                                            'precision': Precision(),
                                            'recall': Recall()
                                        },
                                        device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'accuracy': CategoricalAccuracy(),
                                                'nll': Loss(criterion),
                                                'precision': Precision(),
                                                'recall': Recall(),
                                            },
                                            device=device)

    logger.debug("Setup handlers")
    log_interval = config.get("LOG_INTERVAL", 100)
    reduce_on_plateau = config.get("REDUCE_LR_ON_PLATEAU")

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
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}".format(engine.state.epoch, iter,
                                                                         len(train_loader),
                                                                         engine.state.output[0]))

            writer.add_scalar("training/loss_vs_iterations", engine.state.output[0], engine.state.iteration)

    @trainer.on(Events.EPOCH_STARTED)
    def update_lr_schedulers(engine):
        if lr_schedulers is not None:
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()

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

    log_images_dir = os.path.join(log_dir, "figures")
    os.makedirs(log_images_dir)

    def log_precision_recall_results(engine, epoch, mode):
        for metric_name in ['precision', 'recall']:
            value = engine.state.metrics[metric_name]
            avg_value = torch.mean(value).item()
            writer.add_scalar("{}/avg_{}".format(mode, metric_name), avg_value, epoch)
            # Save metric per class figure
            sorted_values = value.to('cpu').numpy()
            indices = np.argsort(sorted_values)
            sorted_values = sorted_values[indices]
            n_classes = len(sorted_values)
            classes = np.array(["class_{}".format(i) for i in range(n_classes)])
            sorted_classes = classes[indices]
            fig = create_fig_param_per_class(sorted_values, metric_name, classes=sorted_classes, n_classes_per_fig=20)
            fname = os.path.join(log_images_dir, "{}_{}_{}_per_class.png".format(mode, epoch, metric_name))
            fig.savefig(fname)
            # Add figure in TB
            img = Image.open(fname)
            tag = "{}_{}".format(mode, metric_name)
            writer.add_image(tag, np.asarray(img), epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_metrics(engine):
        epoch = engine.state.epoch
        logger.info("One epoch training time (seconds): {}".format(timer.value()))
        logger.info("Training Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}"
                    .format(engine.state.epoch, engine.state.metrics['accuracy'], engine.state.metrics['nll']))
        writer.add_scalar("training/avg_accuracy", engine.state.metrics['accuracy'], epoch)
        writer.add_scalar("training/avg_error", 1.0 - engine.state.metrics['accuracy'], epoch)
        writer.add_scalar("training/avg_loss", engine.state.metrics['nll'], epoch)
        log_precision_recall_results(engine, epoch, "training")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        epoch = engine.state.epoch
        metrics = evaluator.run(val_loader).metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        writer.add_scalar("validation/avg_loss", avg_nll, epoch)
        writer.add_scalar("validation/avg_accuracy", avg_accuracy, epoch)
        writer.add_scalar("validation/avg_error", 1.0 - avg_accuracy, epoch)
        logger.info("Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}"
                    .format(engine.state.epoch, avg_accuracy, avg_nll))
        log_precision_recall_results(evaluator, epoch, "validation")

    if reduce_on_plateau is not None:
        @evaluator.on(Events.COMPLETED)
        def update_reduce_on_plateau(engine):
            val_loss = engine.state.metrics['nll']
            reduce_on_plateau.step(val_loss)

    def score_function(engine):
        val_loss = engine.state.metrics['nll']
        # Objects with highest scores will be retained.
        return -val_loss

    # Setup early stopping:
    if "EARLY_STOPPING_KWARGS" in config:
        kwargs = config["EARLY_STOPPING_KWARGS"]
        if 'score_function' not in kwargs:
            kwargs['score_function'] = score_function
        handler = EarlyStopping(trainer=trainer, **kwargs)
        setup_logger(handler._logger, os.path.join(log_dir, "train.log"), log_level)
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

    n_epochs = config["N_EPOCHS"]
    logger.debug("Start training: {} epochs".format(n_epochs))
    try:
        trainer.run(train_loader, max_epochs=n_epochs)
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

    logger.debug("Training is ended")
    writer.close()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("config_file", type=str, help="Configuration file. See examples in configs/")
    args = parser.parse_args()
    run(args.config_file)

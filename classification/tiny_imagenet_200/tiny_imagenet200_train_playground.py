from __future__ import print_function


import os
import sys
from argparse import ArgumentParser
import random
import logging
from importlib import util

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torchvision.transforms import ToTensor

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engines import Events, create_supervised_evaluator, Engine
from ignite.metrics import CategoricalAccuracy, Loss
from ignite.handlers import ModelCheckpoint, Timer, EarlyStopping
from ignite._utils import to_variable, to_tensor

from dataflow import get_trainval_data_loaders


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
        dummy_input = to_variable(torch.rand(10, 3, 64, 64), cuda=cuda)
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


def save_conf(config_file, logger, writer):
    conf_str = """
        Training configuration file:
        
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
    setup_logger(logger, log_dir, log_level)

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
                                                         num_workers, cuda=cuda)

    optimizer = config["OPTIM"]

    logger.debug("Setup criterion")
    criterion = nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()

    lr_schedulers = config.get("LR_SCHEDULERS")

    def output_transform(output):
        y_pred = output['y_pred']
        y = output['y']
        return to_tensor(y_pred, cpu=not cuda), to_tensor(y, cpu=not cuda)

    logger.debug("Setup ignite trainer and evaluator")
    trainer = create_supervised_trainer(model, optimizer, criterion,
                                        metrics={
                                            'accuracy': CategoricalAccuracy(output_transform=output_transform),
                                            'nll': Loss(criterion, output_transform=output_transform)
                                        },
                                        cuda=cuda)
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'accuracy': CategoricalAccuracy(),
                                                'nll': Loss(criterion)
                                            },
                                            cuda=cuda)

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
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}".format(engine.state.epoch, iter,
                                                                         len(train_loader),
                                                                         engine.state.output['loss']))

            writer.add_scalar("training/loss_vs_iterations", engine.state.output['loss'], engine.state.iteration)

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

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_metrics(engine):

        logger.info("One epoch training time (seconds): {}".format(timer.value()))
        logger.info("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                    .format(engine.state.epoch, engine.state.metrics['accuracy'], engine.state.metrics['nll']))
        writer.add_scalar("training/avg_accuracy", engine.state.metrics['accuracy'], engine.state.epoch)
        writer.add_scalar("training/avg_error", 1.0 - engine.state.metrics['accuracy'], engine.state.epoch)
        writer.add_scalar("training/avg_loss", engine.state.metrics['nll'], engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        metrics = evaluator.run(val_loader).metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        writer.add_scalar("validation/avg_loss", avg_nll, engine.state.epoch)
        writer.add_scalar("validation/avg_accuracy", avg_accuracy, engine.state.epoch)
        writer.add_scalar("validation/avg_error", 1.0 - avg_accuracy, engine.state.epoch)
        logger.info("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                    .format(engine.state.epoch, avg_accuracy, avg_nll))

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

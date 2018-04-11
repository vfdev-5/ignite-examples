from pathlib import Path
import sys
from argparse import ArgumentParser
import random
import logging
from importlib import util


import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engines import Events, Engine
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint, Timer, EarlyStopping
from ignite._utils import to_variable, to_tensor

# Load common module
sys.path.insert(0, Path(__file__).absolute().parent.parent.as_posix())
from common import setup_logger, save_conf


# We need to reimplement ignite.metrics.Loss as
# y and y_pred are lists and not torch.Tensors
class _Loss(Loss):

    def update(self, output):
        y_pred, y = output
        batch_size = y[0].shape[0]
        average_loss = self._loss_fn(to_variable(y_pred, volatile=True), to_variable(y, volatile=True))
        assert average_loss.shape == (1,), '`loss_fn` did not return the average loss'
        self._sum += average_loss.data[0] * batch_size
        self._num_examples += batch_size


def write_model_graph(writer, model, data_loader, cuda):
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    x = to_variable(x, cuda=cuda)
    try:
        writer.add_graph(model, x)
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


def create_supervised_evaluator(model, metrics={}, cuda=False):

    def _prepare_batch(batch):
        x, y, (img_file, target) = batch
        x = to_variable(x, cuda=cuda, volatile=True)
        y = to_variable(y, cuda=cuda, volatile=True)
        return x, y, (img_file, target)

    def _inference(engine, batch):
        model.eval()
        x, y, (img_file, target) = _prepare_batch(batch)
        y_pred = model(x)
        return {
            'y_pred': y_pred,
            'y': y,
            'meta': (img_file, target)
        }

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def load_config(config_filepath):
    assert Path(config_filepath).exists(), "Configuration file '{}' is not found".format(config_filepath)
    # Load custom module
    spec = util.spec_from_file_location("config", config_filepath)
    custom_module = util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    config = custom_module.__dict__
    assert "TRAIN_LOADER" in config, "TRAIN_LOADER parameter is not found in configuration file"
    assert "VAL_LOADER" in config, "TRAIN_LOADER parameter is not found in configuration file"

    assert "OUTPUT_PATH" in config, "OUTPUT_PATH is not found in the configuration file"

    assert "N_EPOCHS" in config, "Number of epochs should be specified in the configuration file"

    assert "MODEL" in config, "MODEL is not found in configuration file"
    if isinstance(config["MODEL"], str) and Path(config["MODEL"]).is_file():
        config["MODEL"] = torch.load(config["MODEL"])
    assert isinstance(config["MODEL"], nn.Module), \
        "Model should be an instance of torch.nn.Module, but given {}".format(type(config["MODEL"]))

    assert "CRITERION" in config, "CRITERION is not found in configuration file"

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
    print("--- VEDAI : Training --- ")

    print("Load config file ... ")
    config = load_config(config_file)

    seed = config.get("SEED", 2018)
    random.seed(seed)
    torch.manual_seed(seed)

    output = Path(config["OUTPUT_PATH"])
    model = config["MODEL"]
    model_name = model.__class__.__name__
    debug = config.get("DEBUG", False)

    from datetime import datetime
    now = datetime.now()
    log_dir = output / ("training_{}_{}".format(model_name, now.strftime("%Y%m%d_%H%M")))
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
        print("Activated debug mode")

    logger = logging.getLogger("VEDAI : Train")
    setup_logger(logger, (log_dir / "train.log").as_posix(), log_level)

    logger.debug("Setup tensorboard writer")
    writer = SummaryWriter(log_dir=(log_dir / "tensorboard").as_posix())

    save_conf(config_file, log_dir.as_posix(), logger, writer)

    cuda = torch.cuda.is_available()
    if cuda:
        logger.debug("CUDA is enabled")
        from torch.backends import cudnn
        cudnn.benchmark = True
        model = model.cuda()

    logger.debug("Setup train/val dataloaders")
    train_loader, val_loader = config["TRAIN_LOADER"], config["VAL_LOADER"]

    write_model_graph(writer, model=model, data_loader=train_loader, cuda=cuda)

    optimizer = config["OPTIM"]

    logger.debug("Setup criterion")
    criterion = config["CRITERION"]
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
                                            'avg_loss': _Loss(criterion, output_transform=output_transform)
                                        },
                                        cuda=cuda)
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'avg_loss': _Loss(criterion, output_transform=output_transform),
                                                # 'avg'
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
        logger.info("Training Results - Epoch: {}  Avg loss: {:.2f}"
                    .format(engine.state.epoch, engine.state.metrics['avg_loss']))
        writer.add_scalar("training/avg_loss", engine.state.metrics['avg_loss'], engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        metrics = evaluator.run(val_loader).metrics
        writer.add_scalar("validation/avg_loss", metrics['avg_loss'], engine.state.epoch)
        logger.info("Validation Results - Epoch: {} Avg loss: {:.2f}"
                    .format(engine.state.epoch, metrics['avg_loss']))

    if reduce_on_plateau is not None:
        @evaluator.on(Events.COMPLETED)
        def update_reduce_on_plateau(engine):
            val_loss = engine.state.metrics['avg_loss']
            reduce_on_plateau.step(val_loss)

    def score_function(engine):
        val_loss = engine.state.metrics['avg_loss']
        # Objects with highest scores will be retained.
        return -val_loss

    # Setup early stopping:
    if "EARLY_STOPPING_KWARGS" in config:
        kwargs = config["EARLY_STOPPING_KWARGS"]
        if 'score_function' not in kwargs:
            kwargs['score_function'] = score_function
        handler = EarlyStopping(trainer=trainer, **kwargs)
        setup_logger(handler._logger, (log_dir / "train.log").as_posix(), log_level)
        evaluator.add_event_handler(Events.COMPLETED, handler)

    # Setup model checkpoint:
    best_model_saver = ModelCheckpoint(log_dir.as_posix(),
                                       filename_prefix="model",
                                       score_name="val_loss",
                                       score_function=score_function,
                                       n_saved=5,
                                       atomic=True,
                                       create_dir=True)
    evaluator.add_event_handler(Events.COMPLETED, best_model_saver, {model_name: model})

    last_model_saver = ModelCheckpoint(log_dir.as_posix(),
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

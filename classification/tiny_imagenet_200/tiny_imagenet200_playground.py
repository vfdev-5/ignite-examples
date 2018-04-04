from __future__ import print_function
import os
from argparse import ArgumentParser

import numpy as np

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, RandomAffine, RandomChoice
from torchvision.transforms import ColorJitter
from torchvision.datasets import ImageFolder
from torchvision.models.squeezenet import squeezenet1_1
import torchvision.utils as vutils

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")

from ignite.engines import Events, create_supervised_evaluator, Engine
from ignite.metrics import CategoricalAccuracy, Loss
from ignite.handlers import ModelCheckpoint, Timer
from ignite._utils import to_variable, to_tensor


def get_data_loaders(dataset_path, train_batch_size, val_batch_size, num_workers, cuda=True):
    data_transform = Compose([
        RandomChoice(
            [
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomAffine(degrees=(-45, 45))
            ]
        ),
        ColorJitter(brightness=0.2, hue=0.2),
        ToTensor(),
        Normalize((-0.5, -0.5, -0.5), (1.0, 1.0, 1.0))
    ])

    trainval_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=data_transform)

    from sklearn.model_selection import StratifiedShuffleSplit

    ssplit = StratifiedShuffleSplit(test_size=0.3, random_state=12345)
    y = np.array([t for _, t in trainval_dataset.samples])
    split_iter = ssplit.split(np.array(trainval_dataset.samples), y)
    train_index, val_index = next(split_iter)

    # Hack to create train/val datasets
    train_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=data_transform)
    train_dataset.samples = np.array(train_dataset.samples, dtype=object)[train_index].tolist()

    val_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=data_transform)
    val_dataset.samples = np.array(val_dataset.samples, dtype=object)[val_index].tolist()

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=cuda)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=cuda)

    return train_loader, val_loader


def create_summary_writer(model, log_dir, cuda):
    writer = SummaryWriter(log_dir=log_dir)
    # try:
    #     dummy_input = to_variable(torch.rand(10, 3, 64, 64), cuda=cuda)
    #     torch.onnx.export(model, dummy_input, "model.proto", verbose=True)
    #     writer.add_graph_onnx("model.proto")
    # except Exception as e:
    #     print("Failed to save model graph: {}".format(e))
    return writer


def get_model(num_classes):
    from torch.nn import Conv2d, AdaptiveAvgPool2d, Sequential
    model = squeezenet1_1(num_classes=num_classes, pretrained=False)
    # As input image size is small 64x64, we modify first layers:
    # replace : Conv2d(3, 64, (3, 3), stride=(2, 2)) by Conv2d(3, 64, (3, 3), stride=(1, 1), padding=1)
    # remove : MaxPool2d (size=(3, 3), stride=(2, 2), dilation=(1, 1)))
    layers = [l for i, l in enumerate(model.features) if i != 2]
    layers[0] = Conv2d(3, 64, kernel_size=(3, 3), padding=1)
    model.features = Sequential(*layers)
    # Replace the last AvgPool2d -> AdaptiveAvgPool2d
    layers = [l for l in model.classifier]
    layers[-1] = AdaptiveAvgPool2d(1)
    model.classifier = Sequential(*layers)
    return model


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


def run(dataset_path, train_batch_size, val_batch_size, num_workers, epochs, lr, log_interval, output, debug):

    from datetime import datetime
    now = datetime.now()
    log_dir = os.path.join(output, "%s" % (now.strftime("%Y%m%d_%H%M")))
    os.makedirs(log_dir, exist_ok=True)

    if debug:
        print("Activated debug mode")

    cuda = torch.cuda.is_available()
    if cuda:
        from torch.backends import cudnn
        cudnn.benchmark = True

    train_loader, val_loader = get_data_loaders(dataset_path, train_batch_size, val_batch_size, num_workers, cuda=cuda)

    model = get_model(num_classes=200)
    model_name = model.__class__.__name__
    if cuda:
        model = model.cuda()

    writer = create_summary_writer(model, os.path.join(log_dir, "tensorboard"), cuda=cuda)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = ExponentialLR(optimizer, gamma=0.98)
    reduce_on_plateau = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

    def output_transform(output):
        y_pred = output['y_pred']
        y = output['y']
        return to_tensor(y_pred, cpu=not cuda), to_tensor(y, cpu=not cuda)

    trainer = create_supervised_trainer(model, optimizer, criterion,
                                        metrics={
                                            'accuracy': CategoricalAccuracy(output_transform=output_transform)
                                        },
                                        cuda=cuda)
    evaluator = create_supervised_evaluator(model,
                                            metrics={
                                                'accuracy': CategoricalAccuracy(),
                                                'nll': Loss(criterion)
                                            },
                                            cuda=cuda)

    # Setup timer to measure training time
    timer = Timer(average=True)
    timer.attach(trainer,
                 start=Events.EPOCH_STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED,
                 step=Events.ITERATION_COMPLETED)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_interval == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output['loss']))
            # if debug:
            #     batch_x, batch_y = engine.state.batch
            #     if batch_y.is_cuda:
            #         batch_y = batch_y.cpu()
            #     m = torch.cat((batch_y.unsqueeze(1), batch_y.unsqueeze(1)), dim=1)
            #     writer.add_embedding(m, metadata=batch_y, label_img=batch_x,
            #                          tag='training data', global_step=engine.state.iteration)
            # if debug:
            #     batch_x, batch_y = engine.state.batch
            #     x = vutils.make_grid(batch_x, scale_each=True, normalize=True)
            #     writer.add_image('training images', x, engine.state.iteration)
            # if debug:
            #     y_preds = engine.state.output['y_preds']
            #     y_probas = to_tensor(F.softmax(y_preds, dim=1), cpu=True)
            #     y_probas = np.argmax(y_probas.numpy(), axis=1)
            #     for p in y_probas:
            #         writer.add_scalar('training y_preds', p, engine.state.iteration)
        writer.add_scalar("training/loss", engine.state.output['loss'], engine.state.iteration)

    @trainer.on(Events.EPOCH_STARTED)
    def update_lr_schedulers(engine):
        lr_scheduler.step()
        lrs = lr_scheduler.get_lr()
        for i, lr in enumerate(lrs):
            writer.add_scalar("learning_rate {}:".format(i), lr, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        writer.add_scalar("training/accuracy", engine.state.metrics['accuracy'], engine.state.epoch)
        print("One interation training time (seconds): {}".format(timer.value()))
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))
        writer.add_scalar("validation/loss", avg_nll, engine.state.epoch)
        writer.add_scalar("validation/accuracy", avg_accuracy, engine.state.epoch)

    @evaluator.on(Events.COMPLETED)
    def update_reduce_on_plateau(engine):
        val_loss = engine.state.metrics['nll']
        reduce_on_plateau.step(val_loss)

    # Setup model checkpoint
    def score_function(engine):
        val_loss = engine.state.metrics['nll']
        # Objects with highest scores will be retained.
        return -val_loss

    handler = ModelCheckpoint(log_dir,
                              filename_prefix="model",
                              score_function=score_function,
                              n_saved=5,
                              atomic=True,
                              create_dir=True,
                              exist_ok=True)
    evaluator.add_event_handler(Events.COMPLETED, handler, {model_name: model})

    trainer.run(train_loader, max_epochs=epochs)

    writer.close()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('dataset_path', type=str,
                        help="Path to Tiny ImageNet dataset. " +
                             "It can be downloaded from : http://cs231n.stanford.edu/tiny-imagenet-200.zip")
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers in data loader(default: 8)')
    parser.add_argument('--val_batch_size', type=int, default=100,
                        help='input batch size for validation (default: 100)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--output", type=str, default="output",
                        help="directory to store best models")
    parser.add_argument("--debug", action="store_true", default=0,
                        help="Debug")

    args = parser.parse_args()

    run(args.dataset_path, args.batch_size, args.val_batch_size, args.num_workers, args.epochs,
        args.lr, args.log_interval, args.output, args.debug)

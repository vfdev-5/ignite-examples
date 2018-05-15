# Basic training configuration file
from pathlib import Path
import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from ignite.engine import Events
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from common.sampling import SmartWeightedRandomSampler
from models.inceptionresnetv2 import FurnitureInceptionResNet299


SEED = 12345
DEBUG = True

OUTPUT_PATH = "output"
DATASET_PATH = Path("/home/fast_storage/imaterialist-challenge-furniture-2018/")

size = 350

TRAIN_TRANSFORMS = [
    RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=3),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(hue=0.12, brightness=0.12),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

VAL_TRANSFORMS = [
    RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=3),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]


BATCH_SIZE = 24
NUM_WORKERS = 15


dataset = FilesFromCsvDataset("output/filtered_train_dataset.csv")
targets = [y - 1 for _, y in dataset]
train_sampler = SmartWeightedRandomSampler(targets)
TRAIN_LOADER = get_data_loader(dataset,
                               data_transform=TRAIN_TRANSFORMS,
                               batch_size=BATCH_SIZE,
                               sampler=train_sampler,
                               num_workers=NUM_WORKERS,
                               pin_memory=True)

val_dataset = FilesFromCsvDataset("output/filtered_val_dataset.csv")
VAL_LOADER = get_data_loader(val_dataset,
                             data_transform=VAL_TRANSFORMS,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             pin_memory=True)


MODEL = FurnitureInceptionResNet299(pretrained='imagenet')

N_EPOCHS = 100

OPTIM = SGD(
    params=[
        {"params": MODEL.stem.parameters(), 'lr': 0.0001},
        {"params": MODEL.features.parameters(), 'lr': 0.002},
        {"params": MODEL.classifier.parameters(), 'lr': 0.1},
    ],
    momentum=0.9)


LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=[4, 8, 12], gamma=0.1)
]


REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=3, threshold=0.08, verbose=True)

EARLY_STOPPING_KWARGS = {
    'patience': 15,
    # 'score_function': None
}

LOG_INTERVAL = 100


TRAINER_CUSTOM_EVENT_HANDLERS = [
    # (event, handler), handler signature should be `foo(trainer, evaluator, logger)`

]


def smart_sampling_update(evaluator, trainer, logger):
    
    if trainer.state.epoch < 5:
        return
    
    assert hasattr(evaluator.state, "metrics"), "Evaluator state has no metrics"
    recall_per_class = evaluator.state.metrics['recall'].cpu()
    mean_recall_per_class = torch.mean(recall_per_class)
    low_recall_classes = np.where(recall_per_class < 0.6 * mean_recall_per_class)[0]
    logger.debug("Smart sampling update: low recall classes (< {}) : {}"
                 .format(mean_recall_per_class, low_recall_classes))
    n_classes = len(recall_per_class)
    class_weights = []
    for c in range(n_classes):
        w = 5.0 if c in low_recall_classes else 1.0
        class_weights.append((c, w))
    train_sampler.update_weights(class_weights)


EVALUATOR_CUSTOM_EVENT_HANDLERS = [
    # (event, handler), handler signature should be `foo(evaluator, trainer, logger)`
    (Events.COMPLETED, smart_sampling_update)
]

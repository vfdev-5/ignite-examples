# Basic training configuration file
from pathlib import Path
import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from common.sampling import get_weighted_train_sampler
from models.inceptionresnetv2 import FurnitureInceptionResNet299


SEED = 765
DEBUG = True

OUTPUT_PATH = "output"
DATASET_PATH = Path("/home/fast_storage/imaterialist-challenge-furniture-2018/")

size = 350

TRAIN_TRANSFORMS = [
    RandomResizedCrop(350, scale=(0.6, 1.0), interpolation=3),
    RandomVerticalFlip(p=0.5),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(hue=0.12, brightness=0.12),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

VAL_TRANSFORMS = [
    RandomResizedCrop(350, scale=(0.7, 1.0), interpolation=3),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]


BATCH_SIZE = 24
NUM_WORKERS = 15


lowest_recall_classes_weight = np.array([
    (3, 7.0), (14, 12.0), (18, 5.0), (26, 5.0), (38, 5.0), (49, 5.0), (62, 10.0), (65, 10.0), (104, 5.0), (123, 5.0)
])

dataset = FilesFromCsvDataset("output/filtered_train_dataset.csv")
train_sampler = get_weighted_train_sampler(dataset, lowest_recall_classes_weight, n_samples=len(dataset))
TRAIN_LOADER = get_data_loader(dataset,
                               data_transform=TRAIN_TRANSFORMS,
                               sampler=train_sampler,
                               batch_size=BATCH_SIZE,
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
        {"params": MODEL.stem.parameters(), 'lr': 0.000095},
        {"params": MODEL.features.parameters(), 'lr': 0.00054},
        {"params": MODEL.classifier.parameters(),
            "lr": 0.095,
            "weight_decay": 0.0001},
    ],
    momentum=0.8,
)


LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=[4, 5, 6, 7, 8, 9, 10, 12], gamma=0.56)
]

# REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=3, threshold=0.08, verbose=True)

EARLY_STOPPING_KWARGS = {
    'patience': 15,
    # 'score_function': None
}

LOG_INTERVAL = 100

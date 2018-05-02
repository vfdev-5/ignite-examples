# Basic training configuration file
import torch
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from models.resnet import FurnitureResNet152_350


SEED = 17
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
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

VAL_TRANSFORMS = [
    RandomResizedCrop(350, scale=(0.7, 1.0), interpolation=3),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]


BATCH_SIZE = 20
NUM_WORKERS = 8

dataset = FilesFromCsvDataset("output/filtered_train_dataset.csv")
TRAIN_LOADER = get_data_loader(dataset,
                               data_transform=TRAIN_TRANSFORMS,
                               batch_size=BATCH_SIZE,
                               num_workers=NUM_WORKERS,
                               cuda=True)

val_dataset = FilesFromCsvDataset("output/filtered_val_dataset.csv")
VAL_LOADER = get_data_loader(val_dataset,
                             data_transform=VAL_TRANSFORMS,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             cuda=True)


model_checkpoint = (Path(OUTPUT_PATH) / "train_resnet152_350_adam_random_crop" / "20180501_2111" /
                    "model_FurnitureResNet152_350_6_val_loss=0.5574015.pth").as_posix()
MODEL = torch.load(model_checkpoint)


N_EPOCHS = 100

OPTIM = Adam(
    params=[
        {"params": MODEL.stem.parameters(), 'lr': 0.0000063},
        {"params": MODEL.features.parameters(), 'lr': 0.0000063},
        {"params": MODEL.classifier.parameters(), 'lr': 0.000063},
    ],
)


LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=[1, 2, 3, 4, 5, 6, 7, 8], gamma=0.55)
]

EARLY_STOPPING_KWARGS = {
    'patience': 15,
    # 'score_function': None
}

LOG_INTERVAL = 100

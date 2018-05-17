# Basic training configuration file
from pathlib import Path
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from models.inceptionv4 import FurnitureInceptionV4_350

SEED = 12345
DEBUG = True

OUTPUT_PATH = Path("output") / "trains"

size = 350

TRAIN_TRANSFORMS = [
    RandomResizedCrop(size, scale=(0.6, 1.0), interpolation=3),
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


BATCH_SIZE = 48
NUM_WORKERS = 15


# Classify only low recall classes + missed classified : 14, 22, 28, 31, 39, 56, 62, 101, 125
hard_examples = [14, 22, 28, 31, 39, 56, 62, 101, 125]
he_map = dict([(v, i) for i, v in enumerate(hard_examples)])


def hard_examples_only(target):
    return he_map[target] if target in he_map else len(hard_examples)


dataset = FilesFromCsvDataset("output/unique_filtered_train_dataset.csv")

TRAIN_LOADER = get_data_loader(dataset,
                               data_transform=TRAIN_TRANSFORMS,
                               target_transform=hard_examples_only,
                               batch_size=BATCH_SIZE,
                               num_workers=NUM_WORKERS,
                               pin_memory=True)

val_dataset = FilesFromCsvDataset("output/unique_filtered_val_dataset.csv")
VAL_LOADER = get_data_loader(val_dataset,
                             data_transform=VAL_TRANSFORMS,
                             target_transform=hard_examples_only,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             pin_memory=True)


MODEL = FurnitureInceptionV4_350(pretrained='imagenet')

N_EPOCHS = 100

OPTIM = SGD(
    params=[
        {"params": MODEL.stem.parameters(), 'lr': 0.0004},
        {"params": MODEL.features.parameters(), 'lr': 0.0009},
        {"params": MODEL.classifier.parameters(), 'lr': 0.09},
    ],
    momentum=0.9)


LR_SCHEDULERS = [
    ExponentialLR(OPTIM, gamma=0.9),
]

REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=3, threshold=0.1, verbose=True)

EARLY_STOPPING_KWARGS = {
    'patience': 15,
    # 'score_function': None
}

LOG_INTERVAL = 100

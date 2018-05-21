# Basic training configuration file
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from models.inceptionv4 import FurnitureInceptionV4_350


SEED = 57
DEBUG = True
DEVICE = 'cuda'

OUTPUT_PATH = Path("output") / "train"


size = 350

TRAIN_TRANSFORMS = [
    RandomResizedCrop(size, scale=(0.6, 1.0), interpolation=3),
    RandomVerticalFlip(p=0.5),
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


BATCH_SIZE = 36
NUM_WORKERS = 15


dataset = FilesFromCsvDataset("output/unique_filtered_train_dataset.csv")
TRAIN_LOADER = get_data_loader(dataset,
                               data_transform=TRAIN_TRANSFORMS,
                               batch_size=BATCH_SIZE,
                               num_workers=NUM_WORKERS,
                               pin_memory='cuda' in DEVICE)


val_dataset = FilesFromCsvDataset("output/unique_filtered_val_dataset.csv")
VAL_LOADER = get_data_loader(val_dataset,
                             data_transform=VAL_TRANSFORMS,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             pin_memory='cuda' in DEVICE)


MODEL = FurnitureInceptionV4_350(pretrained='imagenet')


hard_examples = [14, 62, 65, 123, 18, 3, 50, 104, 26, 22, 96, 109, 48, 49, 69]
weights = list([1.0] * 128)

for i, c in enumerate(hard_examples):
    weights[c] = 3.0 if i < 5 else 2.0

loss_weights = torch.tensor(weights)
CRITERION = CrossEntropyLoss(weight=loss_weights)


N_EPOCHS = 100


OPTIM = Adam(
    params=[
        {"params": MODEL.stem.parameters(), 'lr': 0.0001},
        {"params": MODEL.features.parameters(), 'lr': 0.0001},
        {"params": MODEL.classifier.parameters(), 'lr': 0.002},
    ],
)


LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=[4, 5, 6, 8, 9, 10, 12], gamma=0.2)
]


REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=3, threshold=0.1, verbose=True)


EARLY_STOPPING_KWARGS = {
    'patience': 15,
    # 'score_function': None
}

LOG_INTERVAL = 100

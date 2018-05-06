# Basic training configuration file
from pathlib import Path
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, RandomCrop, CenterCrop
from torchvision.transforms import RandomApply, RandomAffine
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import get_data_loaders
from models.dpn import FurnitureDPN131_350

SEED = 12345
DEBUG = True

OUTPUT_PATH = "output"
DATASET_PATH = Path("/home/fast_storage/imaterialist-challenge-furniture-2018/")

size = 350

TRAIN_TRANSFORMS = [
    RandomApply(
        [RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.7, 1.2), resample=2), ],
        p=0.5
    ),
    RandomCrop(size=size),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ColorJitter(hue=0.1, brightness=0.1),
    ToTensor(),
    Normalize(mean=[124 / 255, 117 / 255, 104 / 255], std=[1 / (.0167 * 255)] * 3)
]
VAL_TRANSFORMS = [
    CenterCrop(size=size),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[124 / 255, 117 / 255, 104 / 255], std=[1 / (.0167 * 255)] * 3)
]


BATCH_SIZE = 10
NUM_WORKERS = 8

TRAIN_LOADER, VAL_LOADER = get_data_loaders(
    train_dataset_path=DATASET_PATH / "train_400x400",
    val_dataset_path=DATASET_PATH / "val_400x400",
    train_data_transform=TRAIN_TRANSFORMS,
    val_data_transform=VAL_TRANSFORMS,
    train_batch_size=BATCH_SIZE,
    val_batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True)


MODEL = FurnitureDPN131_350(pretrained='imagenet')

N_EPOCHS = 100

OPTIM = SGD(
    params=[
        {"params": MODEL.stem.parameters(), 'lr': 0.0005},
        {"params": MODEL.features.parameters(), 'lr': 0.001},
        {"params": MODEL.classifier.parameters(), 'lr': 0.1},
    ],
    momentum=0.9)


LR_SCHEDULERS = [
    ExponentialLR(OPTIM, gamma=0.99)
]

REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=5, threshold=0.05, verbose=True)

EARLY_STOPPING_KWARGS = {
    'patience': 30,
    # 'score_function': None
}

LOG_INTERVAL = 100

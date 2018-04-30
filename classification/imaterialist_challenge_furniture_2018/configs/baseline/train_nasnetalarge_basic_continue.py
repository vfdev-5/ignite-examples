# Basic training configuration file
from pathlib import Path
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, RandomCrop, CenterCrop
from torchvision.transforms import RandomApply, RandomAffine
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import get_data_loaders
# from models.nasnet_a_large import FurnitureNASNetALarge

SEED = 12345
DEBUG = True

OUTPUT_PATH = "output"
DATASET_PATH = Path("/home/fast_storage/imaterialist-challenge-furniture-2018/")


TRAIN_TRANSFORMS = [
    RandomApply(
        [RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.7, 1.2), resample=2), ],
        p=0.5
    ),
    RandomCrop(size=331),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ColorJitter(hue=0.1, brightness=0.1),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]
VAL_TRANSFORMS = [
    CenterCrop(size=331),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]


BATCH_SIZE = 8
NUM_WORKERS = 8

TRAIN_LOADER, VAL_LOADER = get_data_loaders(
    train_dataset_path=DATASET_PATH / "train_400x400",
    val_dataset_path=DATASET_PATH / "val_400x400",
    train_data_transform=TRAIN_TRANSFORMS,
    val_data_transform=VAL_TRANSFORMS,
    train_batch_size=BATCH_SIZE,
    val_batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    cuda=True)


# MODEL = FurnitureNASNetALarge(pretrained='imagenet')
model_checkpoint = (Path(OUTPUT_PATH) / "training_FurnitureNASNetALarge_20180416_0626" /
                    "model_FurnitureNASNetALarge_2_val_loss=0.5761203.pth").as_posix()
MODEL = torch.load(model_checkpoint)

N_EPOCHS = 100

OPTIM = SGD(
    params=[
        {"params": MODEL.stem.parameters(), 'lr': 0.0002},
        {"params": MODEL.features.parameters(), 'lr': 0.0007},
        {"params": MODEL.classifier.parameters(), 'lr': 0.07},
    ],
    momentum=0.9)

LR_SCHEDULERS = [
    ExponentialLR(OPTIM, gamma=0.8)
]

LOG_INTERVAL = 100

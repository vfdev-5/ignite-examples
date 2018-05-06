# Basic training configuration file
from pathlib import Path
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, RandomCrop, CenterCrop
from torchvision.transforms import RandomApply, RandomAffine
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import get_data_loaders
from models.nasnet_a_large import FurnitureNASNetALarge

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
    pin_memory=True)


MODEL = FurnitureNASNetALarge(pretrained='imagenet')

N_EPOCHS = 100

OPTIM = SGD(
    params=[
        {"params": MODEL.stem.parameters(), 'lr': 0.0005},
        {"params": MODEL.features.parameters(), 'lr': 0.001},
        {"params": MODEL.classifier.parameters(), 'lr': 0.1},
    ],
    momentum=0.9)

# LR_SCHEDULERS = [
#     MultiStepLR(OPTIM, milestones=[55, 70, 80, 90, 100], gamma=0.5)
# ]

REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=5, threshold=0.07, verbose=True)

EARLY_STOPPING_KWARGS = {
    'patience': 15,
    # 'score_function': None
}

LOG_INTERVAL = 100

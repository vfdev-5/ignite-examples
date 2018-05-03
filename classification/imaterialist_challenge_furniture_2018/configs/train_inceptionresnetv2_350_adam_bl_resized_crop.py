# Basic training configuration file
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from models.inceptionresnetv2 import FurnitureInceptionResNet299
from common.boostrapping_loss import SoftBootstrappingLoss


SEED = 12345
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

CRITERION = SoftBootstrappingLoss(beta=0.95)

MODEL = FurnitureInceptionResNet299(pretrained='imagenet')

N_EPOCHS = 100

OPTIM = Adam(
    params=[
        {"params": MODEL.stem.parameters(), 'lr': 0.00005},
        {"params": MODEL.features.parameters(), 'lr': 0.0001},
        {"params": MODEL.classifier.parameters(), 'lr': 0.001},
    ],
)


LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=[4, 5, 6, 8, 10, 12], gamma=0.2)
]


REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=3, threshold=0.08, verbose=True)

EARLY_STOPPING_KWARGS = {
    'patience': 15,
    # 'score_function': None
}

LOG_INTERVAL = 100

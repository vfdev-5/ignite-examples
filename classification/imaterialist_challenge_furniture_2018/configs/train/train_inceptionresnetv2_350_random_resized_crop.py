# Basic training configuration file
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop, RandomChoice
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from models.inceptionresnetv2 import FurnitureInceptionResNet299


SEED = 17
DEBUG = True

OUTPUT_PATH = "output"

size = 350

TRAIN_TRANSFORMS = [
    RandomChoice(
        [
            RandomResizedCrop(size, scale=(0.3, 8.0), interpolation=3),
            RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=3),
        ]
    ),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(hue=0.12, brightness=0.12),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

VAL_TRANSFORMS = [
    RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=3),
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
        {"params": MODEL.stem.parameters(), 'lr': 0.01},
        {"params": MODEL.features.parameters(), 'lr': 0.07},
        {"params": MODEL.classifier.parameters(), 'lr': 0.1},
    ]
)

REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=4, threshold=0.1, verbose=True)

EARLY_STOPPING_KWARGS = {
    'patience': 25,
    # 'score_function': None
}

LOG_INTERVAL = 100

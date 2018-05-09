# Basic training configuration file
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop, RandomChoice
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from models.inceptionv4 import FurnitureInceptionV4_350_FC


SEED = 42
DEBUG = True
DEVICE = 'cuda'

OUTPUT_PATH = "output"

size = 350

TRAIN_TRANSFORMS = [
    RandomChoice(
        [
            RandomResizedCrop(size, scale=(0.6, 8.0), interpolation=3),
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
                               pin_memory='cuda' in DEVICE)


val_dataset = FilesFromCsvDataset("output/filtered_val_dataset.csv")
VAL_LOADER = get_data_loader(val_dataset,
                             data_transform=VAL_TRANSFORMS,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             pin_memory='cuda' in DEVICE)


MODEL = FurnitureInceptionV4_350_FC(pretrained='imagenet')


N_EPOCHS = 100


OPTIM = RMSprop(
    params=[
        {"params": MODEL.stem.parameters(), 'lr': 0.0012},
        {"params": MODEL.features.parameters(), 'lr': 0.0034},
        {"params": MODEL.classifier.parameters(), 'lr': 0.0045},
        {"params": MODEL.final_classifier.parameters(), 'lr': 0.045},
    ],
    alpha=0.9,
    eps=1.0)


LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=[2, 4, 5, 6, 7, 8, 9, 10, 11, 12], gamma=0.92)
]

# REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=2, threshold=0.1, verbose=True)


EARLY_STOPPING_KWARGS = {
    'patience': 15,
    # 'score_function': None
}


LOG_INTERVAL = 100

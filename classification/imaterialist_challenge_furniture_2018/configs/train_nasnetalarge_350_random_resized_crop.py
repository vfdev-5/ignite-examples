# Basic training configuration file
from torch.optim import RMSprop
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop, RandomChoice
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from models.nasnet_a_large import FurnitureNASNetALarge350


SEED = 17
DEBUG = True
DEVICE = 'cuda'

OUTPUT_PATH = "output"

size = 350

TRAIN_TRANSFORMS = [
    RandomChoice(
        [
            RandomResizedCrop(size, scale=(0.5, 8.0), interpolation=3),
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


BATCH_SIZE = 8
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


MODEL = FurnitureNASNetALarge350(pretrained='imagenet')

N_EPOCHS = 15


OPTIM = RMSprop(
    params=[
        {"params": MODEL.stem.parameters(), 'lr': 0.0012},
        {"params": MODEL.features.parameters(), 'lr': 0.004},
        {"params": MODEL.classifier.parameters(), 'lr': 0.045},
    ],
    alpha=0.9,
    eps=1.0
)


def lambda_lr_stem(epoch):
    return 0.5 ** epoch


def lambda_lr_features(epoch):
    return 0.65 ** epoch


def lambda_lr_classifier(epoch):
    return 0.8 ** epoch


LR_SCHEDULERS = [
    LambdaLR(OPTIM, lr_lambda=[lambda_lr_stem, lambda_lr_features, lambda_lr_classifier])
]


EARLY_STOPPING_KWARGS = {
    'patience': 5,
    # 'score_function': None
}

LOG_INTERVAL = 100

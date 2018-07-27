# Basic training configuration file
from torch.optim import RMSprop
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop, RandomChoice
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from common.lr_schedulers import LRSchedulerWithRestart
from models.senet import FurnitureSENet154_350


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
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

VAL_TRANSFORMS = [
    RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=3),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]


BATCH_SIZE = 10
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


MODEL = FurnitureSENet154_350(pretrained='imagenet')


N_EPOCHS = 100


OPTIM = RMSprop(
    params=[
        {"params": MODEL.stem.parameters(), 'lr': 0.0012},
        {"params": MODEL.features.parameters(), 'lr': 0.004},
        {"params": MODEL.classifier.parameters(), 'lr': 0.045},
    ],
    alpha=0.9,
    eps=1.0
)


lr_scheduler = MultiStepLR(OPTIM, milestones=[2, 4, 5, 6, 7, 8, 9, 10, 11, 12], gamma=0.7)
lr_scheduler_restarts = LRSchedulerWithRestart(lr_scheduler,
                                               restart_every=15,
                                               restart_factor=1.0,
                                               init_lr_factor=0.7)

LR_SCHEDULERS = [
    lr_scheduler_restarts
]

# REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=2, threshold=0.1, verbose=True)


EARLY_STOPPING_KWARGS = {
    'patience': 15,
    # 'score_function': None
}


LOG_INTERVAL = 100

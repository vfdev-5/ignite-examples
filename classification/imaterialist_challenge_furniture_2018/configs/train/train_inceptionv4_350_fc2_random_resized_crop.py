# Basic training configuration file
from pathlib import Path
from torch.optim import RMSprop
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, Compose
from torchvision.transforms import RandomResizedCrop, RandomAffine, RandomApply, RandomChoice
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from models.inceptionv4 import FurnitureInceptionV4_350_FC2


SEED = 43
DEBUG = True
DEVICE = 'cuda'

OUTPUT_PATH = Path("output") / "train"

size = 350

TRAIN_TRANSFORMS = Compose([
    RandomApply(
        [RandomAffine(degrees=15, resample=3, fillcolor=(255, 255, 255)), ],
        p=0.5
    ),
    RandomChoice(
        [
            RandomResizedCrop(size, scale=(0.4, 0.6), interpolation=3),
            RandomResizedCrop(size, scale=(0.6, 1.0), interpolation=3),
        ]
    ),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(hue=0.12, brightness=0.12),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


VAL_TRANSFORMS = TRAIN_TRANSFORMS


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


MODEL = FurnitureInceptionV4_350_FC2(pretrained='imagenet')


N_EPOCHS = 100


OPTIM = RMSprop(
    params=[
        {"params": MODEL.stem.parameters(), 'lr': 0.0015},
        {"params": MODEL.features.parameters(), 'lr': 0.0034},
        {"params": MODEL.classifier.parameters(), 'lr': 0.0045},
        {"params": MODEL.final_classifier.parameters(), 'lr': 0.045},
    ],
    alpha=0.9,
    eps=1.0)


LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=[2, 4, 5, 6, 8, 9, 10, 11, 13], gamma=0.92)
]


EARLY_STOPPING_KWARGS = {
    'patience': 15,
    # 'score_function': None
}


LOG_INTERVAL = 100

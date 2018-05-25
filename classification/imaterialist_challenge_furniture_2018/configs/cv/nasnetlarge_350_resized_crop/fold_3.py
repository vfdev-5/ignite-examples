# Basic training configuration file
from pathlib import Path
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import ConcatDataset
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import ColorJitter, ToTensor, Normalize

from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader, get_trainval_indices
from models.nasnet_a_large import FurnitureNASNetALarge350


SEED = 1245
DEBUG = True
DEVICE = "cuda"

OUTPUT_PATH = Path("output") / "cv" / "nasnetlarge_350_resized_crop"

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


batch_size = 8
num_workers = 15


train_dataset = FilesFromCsvDataset("output/unique_filtered_train_dataset.csv")
val_dataset = FilesFromCsvDataset("output/unique_filtered_val_dataset.csv")
trainval_dataset = ConcatDataset([train_dataset, val_dataset])


# #### Stratified split :
fold_index = 3
n_splits = 4
train_index, val_index = get_trainval_indices(trainval_dataset,
                                              fold_index=fold_index, n_splits=n_splits,
                                              xy_transforms=None,
                                              batch_size=batch_size, n_workers=8,
                                              seed=SEED)
# ####


TRAIN_LOADER = get_data_loader(trainval_dataset,
                               data_transform=TRAIN_TRANSFORMS,
                               sample_indices=train_index,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               pin_memory="cuda" in DEVICE)


VAL_LOADER = get_data_loader(trainval_dataset,
                             data_transform=VAL_TRANSFORMS,
                             sample_indices=val_index,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_memory="cuda" in DEVICE)


MODEL = FurnitureNASNetALarge350(pretrained='imagenet')


N_EPOCHS = 10


OPTIM = SGD(
    params=[
        {"params": MODEL.stem.parameters(), 'lr': 0.0001},
        {"params": MODEL.features.parameters(), 'lr': 0.01},
        {"params": MODEL.classifier.parameters(), 'lr': 0.1},
    ],
    momentum=0.9)


LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=[2, 4, 6, 8, 10, 12], gamma=0.15)
]


REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=3, threshold=0.08, verbose=True)

EARLY_STOPPING_KWARGS = {
    'patience': 15,
    # 'score_function': None
}

LOG_INTERVAL = 100

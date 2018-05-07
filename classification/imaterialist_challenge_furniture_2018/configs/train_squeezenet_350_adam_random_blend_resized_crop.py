# Basic training configuration file
from pathlib import Path
from functools import partial
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import ColorJitter, ToTensor, Normalize, Compose
from common.dataset import FilesFromCsvDataset, TransformedDataset, read_image
from common.data_loaders import get_data_loader
from common.transforms import RandomMultiImageAugDataset, basic_random_half_blend
from models.squeezenet_350 import FurnitureSqueezeNet350


SEED = 12345
DEBUG = True

OUTPUT_PATH = "output"
DATASET_PATH = Path("/home/fast_storage/imaterialist-challenge-furniture-2018/")


TRAIN_TRANSFORMS = [
    RandomResizedCrop(350, scale=(0.7, 1.0)),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ColorJitter(hue=0.1, brightness=0.2, contrast=0.2),
]

VAL_TRANSFORMS = [
    RandomResizedCrop(350, scale=(0.8, 1.0)),
    RandomHorizontalFlip(p=0.5),
]

common_transform = [
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]


BATCH_SIZE = 180
NUM_WORKERS = 15

n_classes = 128
dataset = FilesFromCsvDataset("output/filtered_train_dataset.csv")
dataset = TransformedDataset(dataset, transforms=read_image, target_transforms=lambda l: l - 1)
dataset = TransformedDataset(dataset, transforms=Compose(TRAIN_TRANSFORMS))
dataset = RandomMultiImageAugDataset(dataset, n_classes, aug_fn=partial(basic_random_half_blend, alpha=0.3))
dataset = TransformedDataset(dataset, transforms=Compose(common_transform))

TRAIN_LOADER = get_data_loader(dataset,
                               data_transform=None,
                               batch_size=BATCH_SIZE,
                               num_workers=NUM_WORKERS,
                               pin_memory=True)

val_dataset = FilesFromCsvDataset("output/filtered_val_dataset.csv")
VAL_LOADER = get_data_loader(val_dataset,
                             data_transform=VAL_TRANSFORMS + common_transform,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             pin_memory=True)

MODEL = FurnitureSqueezeNet350(pretrained=True)

N_EPOCHS = 100

OPTIM = Adam(
    params=[
        {"params": MODEL.features.parameters(), 'lr': 0.0001},
        {"params": MODEL.classifier.parameters(), 'lr': 0.001},
    ],
)

LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=[5, 7, 9, 10, 11, 12, 13], gamma=0.5)
]

REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=5, threshold=0.05, verbose=True)

EARLY_STOPPING_KWARGS = {
    'patience': 30,
    # 'score_function': None
}

LOG_INTERVAL = 100

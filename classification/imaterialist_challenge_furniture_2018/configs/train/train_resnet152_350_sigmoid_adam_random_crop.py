# Basic training configuration file
from pathlib import Path
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from ignite._utils import to_onehot
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from models.resnet import FurnitureResNet152_350


SEED = 17
DEBUG = True
DEVICE = "cuda"

OUTPUT_PATH = Path("output") / "train"


size = 350

TRAIN_TRANSFORMS = [
    RandomResizedCrop(size, scale=(0.6, 1.0), interpolation=3),
    RandomVerticalFlip(p=0.5),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(hue=0.12, brightness=0.12),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

VAL_TRANSFORMS = [
    RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=3),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]


BATCH_SIZE = 20
NUM_WORKERS = 8

dataset = FilesFromCsvDataset("output/unique_filtered_train_dataset.csv")
TRAIN_LOADER = get_data_loader(dataset,
                               data_transform=TRAIN_TRANSFORMS,
                               batch_size=BATCH_SIZE,
                               num_workers=NUM_WORKERS,
                               pin_memory=True)

val_dataset = FilesFromCsvDataset("output/unique_filtered_val_dataset.csv")
VAL_LOADER = get_data_loader(val_dataset,
                             data_transform=VAL_TRANSFORMS,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             pin_memory=True)


MODEL = FurnitureResNet152_350(pretrained='imagenet')


n_classes = 128


class _BCEWithLogitsLoss(BCEWithLogitsLoss):

    def forward(self, input, target):
        target = to_onehot(target, num_classes=n_classes).float()
        return super(_BCEWithLogitsLoss, self).forward(input, target)


CRITERION = _BCEWithLogitsLoss(size_average=False)

N_EPOCHS = 15

OPTIM = Adam(
    params=[
        {"params": MODEL.stem.parameters(), 'lr': 0.0001},
        {"params": MODEL.features.parameters(), 'lr': 0.0001},
        {"params": MODEL.classifier.parameters(), 'lr': 0.001},
    ],
    amsgrad=True
)


LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=[4, 8, 12, 13, 14], gamma=0.5)
]

EARLY_STOPPING_KWARGS = {
    'patience': 15,
    # 'score_function': None
}

LOG_INTERVAL = 100

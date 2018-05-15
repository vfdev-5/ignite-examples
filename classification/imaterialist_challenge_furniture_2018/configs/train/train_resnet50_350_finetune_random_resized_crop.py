# Basic training configuration file
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import RandomResizedCrop, RandomChoice
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from models.resnet import FurnitureResNet50_350_finetune


SEED = 17
DEBUG = True
DEVICE = 'cuda'

OUTPUT_PATH = "output"


size = 350

TRAIN_TRANSFORMS = [
    RandomChoice(
        [
            RandomResizedCrop(size, scale=(0.4, 6.0), interpolation=3),
            RandomResizedCrop(size, scale=(0.6, 1.0), interpolation=3),
        ]
    ),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
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


BATCH_SIZE = 64
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


MODEL = FurnitureResNet50_350_finetune(pretrained='imagenet')


N_EPOCHS = 100


OPTIM = Adam(
    params=[
        {"params": MODEL.features.parameters(), 'lr': 0.001},
        {"params": MODEL.classifier.parameters(), 'lr': 0.001},
        {"params": MODEL.final_classifiers.parameters(), 'lr': 0.001},
    ],
)


def lambda_lr_features(epoch):
    if epoch < 5:
        return 0.001
    else:
        return 0.1 * (0.7 ** (epoch - 3))


def lambda_lr_classifier(epoch):
    if epoch < 5:
        return 0.01
    else:
        return 0.7 ** (epoch - 3)


def lambda_lr_final_classifiers(epoch):
    if epoch < 5:
        return 1.0
    else:
        return 0.75 ** (epoch - 3)


LR_SCHEDULERS = [
    LambdaLR(OPTIM, lr_lambda=[lambda_lr_features, lambda_lr_classifier, lambda_lr_final_classifiers])
]

EARLY_STOPPING_KWARGS = {
    'patience': 15,
}


LOG_INTERVAL = 100

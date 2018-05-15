# Basic training configuration file
import torch
from torch.optim import RMSprop
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, Compose, RandomResizedCrop
from torchvision.transforms import FiveCrop, Lambda, Resize
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from models.inceptionresnetv2 import FurnitureInceptionResNetOnFiveCrops


SEED = 17
DEBUG = True
DEVICE = 'cuda'

OUTPUT_PATH = "output"

size = 224

single_img_augs = Compose([
    RandomHorizontalFlip(p=0.5),
    ColorJitter(hue=0.12, brightness=0.12),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

TRAIN_TRANSFORMS = Compose([
    Resize(int(size * 1.5), interpolation=2),
    FiveCrop(size=size),
    Lambda(lambda crops: torch.stack([single_img_augs(crop) for crop in crops]))
])

VAL_TRANSFORMS = TRAIN_TRANSFORMS


BATCH_SIZE = 7
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


MODEL = FurnitureInceptionResNetOnFiveCrops(pretrained='imagenet', n_cls_layers=176)

for param in MODEL.stem.parameters():
    param.requires_grad = False

for param in MODEL.low_features.parameters():
    param.requires_grad = False


N_EPOCHS = 100


OPTIM = RMSprop(
    params=[
        # {"params": MODEL.stem.parameters(), 'lr': 0.00045},
        # {"params": MODEL.low_features.parameters(), 'lr': 0.0045},
        {"params": MODEL.features.parameters(), 'lr': 0.0045},
        {"params": MODEL.crop_classifiers.parameters(), 'lr': 0.045},
        {"params": MODEL.final_classifier.parameters(), 'lr': 0.045},
    ],
    alpha=0.9,
    eps=1.0)


LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=[2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20], gamma=0.94)
]


EARLY_STOPPING_KWARGS = {
    'patience': 25,
    # 'score_function': None
}


LOG_INTERVAL = 100

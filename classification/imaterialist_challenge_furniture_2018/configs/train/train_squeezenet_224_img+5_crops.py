# Basic training configuration file
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, Compose, RandomResizedCrop
from torchvision.transforms import FiveCrop, Lambda, Resize
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from models.squeezenet_350 import FurnitureSqueezeNetOnCrops


SEED = 12
DEBUG = True
DEVICE = 'cuda'

OUTPUT_PATH = "output"


single_img_augs = Compose([
    RandomHorizontalFlip(p=0.5),
    ColorJitter(hue=0.12, brightness=0.12),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

size = 224
augs_branch_1 = RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=2)
augs_branch_2 = Compose([Resize(420, interpolation=2), FiveCrop(size=size)])


TRAIN_TRANSFORMS = Compose([
    Lambda(lambda img: (augs_branch_1(img), ) + augs_branch_2(img)),
    Lambda(lambda crops: torch.stack([single_img_augs(crop) for crop in crops]))
])

VAL_TRANSFORMS = TRAIN_TRANSFORMS


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


MODEL = FurnitureSqueezeNetOnCrops(pretrained=True)


N_EPOCHS = 100


OPTIM = Adam(
    params=[
        {"params": MODEL.features.parameters(), 'lr': 0.0001},
        {"params": MODEL.crop_classifiers.parameters(), 'lr': 0.001},
        {"params": MODEL.final_classifier.parameters(), 'lr': 0.002},
    ],
)

LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=list(range(5, 50, 2)), gamma=0.8)
]

# REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=5, threshold=0.05, verbose=True)

EARLY_STOPPING_KWARGS = {
    'patience': 30,
    # 'score_function': None
}

LOG_INTERVAL = 100

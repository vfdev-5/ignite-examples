# Basic training configuration file
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, Compose, RandomResizedCrop
from torchvision.transforms import FiveCrop, Lambda, Resize
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from models.model_on_crops import FurnitureModelOnCrops
from pretrainedmodels.models.inceptionv4 import inceptionv4


SEED = 12
DEBUG = True
DEVICE = 'cuda'

OUTPUT_PATH = "output"


single_img_augs = Compose([
    RandomHorizontalFlip(p=0.5),
    ColorJitter(hue=0.12, brightness=0.12),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

size = 180

augs_branch_1 = RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=2)
augs_branch_2 = Compose([Resize(int(1.9 * size), interpolation=2), FiveCrop(size=size)])


TRAIN_TRANSFORMS = Compose([
    Lambda(lambda img: (augs_branch_1(img), ) + augs_branch_2(img)),
    Lambda(lambda crops: torch.stack([single_img_augs(crop) for crop in crops]))
])

VAL_TRANSFORMS = TRAIN_TRANSFORMS


BATCH_SIZE = 24
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


base_model = inceptionv4(num_classes=1000, pretrained='imagenet')
MODEL = FurnitureModelOnCrops(features=base_model.features, featuremap_output_size=1536, n_cls_layers=1024)


N_EPOCHS = 100


OPTIM = Adam(
    params=[
        {"params": MODEL.base_features.parameters(), 'lr': 0.0001},
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

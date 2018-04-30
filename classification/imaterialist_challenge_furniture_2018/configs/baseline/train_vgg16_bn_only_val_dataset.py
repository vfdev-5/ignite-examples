# Basic training configuration file
from pathlib import Path
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, RandomCrop, CenterCrop
from torchvision.transforms import RandomApply, RandomAffine
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.data_loaders import get_trainval_data_loaders, get_trainval_indices
from common.dataset import TrainvalFilesDataset
from models.vgg import FurnitureVGG16BN

SEED = 12345
DEBUG = True

OUTPUT_PATH = "output"
DATASET_PATH = Path("/home/fast_storage/imaterialist-challenge-furniture-2018/")


TRAIN_TRANSFORMS = [
    RandomApply(
        [RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.7, 1.2), resample=2), ],
        p=0.5
    ),
    RandomCrop(size=224),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ColorJitter(hue=0.1, brightness=0.1),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
VAL_TRANSFORMS = [
    CenterCrop(size=224),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]


BATCH_SIZE = 32
NUM_WORKERS = 8


dataset = TrainvalFilesDataset(DATASET_PATH / "val_400x400")
train_index, val_index = get_trainval_indices(dataset,
                                              fold_index=0, n_splits=5,
                                              batch_size=BATCH_SIZE, n_workers=NUM_WORKERS)

TRAIN_LOADER, VAL_LOADER = get_trainval_data_loaders(dataset, train_index, val_index,
                                                     train_transforms=TRAIN_TRANSFORMS,
                                                     val_transforms=VAL_TRANSFORMS,
                                                     train_batch_size=BATCH_SIZE,
                                                     val_batch_size=BATCH_SIZE,
                                                     num_workers=NUM_WORKERS, pin_memory=True)

MODEL = FurnitureVGG16BN(pretrained=True)

N_EPOCHS = 100

OPTIM = SGD(
    params=[
        {"params": MODEL.features.parameters(), 'lr': 0.0001},
        {"params": MODEL.classifier.parameters(), 'lr': 0.1},
    ])

# LR_SCHEDULERS = [
#     MultiStepLR(OPTIM, milestones=[55, 70, 80, 90, 100], gamma=0.5)
# ]

REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=5, threshold=0.05, verbose=True)

EARLY_STOPPING_KWARGS = {
    'patience': 30,
    # 'score_function': None
}

LOG_INTERVAL = 100

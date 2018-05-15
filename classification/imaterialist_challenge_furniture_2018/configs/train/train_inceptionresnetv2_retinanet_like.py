# Basic training configuration file
from pathlib import Path
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, Compose
from torchvision.transforms import RandomResizedCrop, RandomAffine, RandomApply
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from models.retinanet_cls_only import FurnitureRetinaNetClassification


SEED = 17
DEBUG = True
DEVICE = 'cuda'

OUTPUT_PATH = Path("output") / "train"

size = 350

TRAIN_TRANSFORMS = Compose([
    RandomApply(
        [RandomAffine(degrees=15, resample=3, fillcolor=(255, 255, 255)), ],
        p=0.5
    ),
    RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=3),
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


MODEL = FurnitureRetinaNetClassification(num_classes=128, pretrained=True)


N_EPOCHS = 100


OPTIM = Adam(
    params=[
        {"params": MODEL.fpn.stem.parameters(), 'lr': 0.0001},
        {"params": MODEL.fpn.low_features.parameters(), 'lr': 0.00015},
        {"params": MODEL.fpn.mid_features.parameters(), 'lr': 0.00015},
        {"params": MODEL.fpn.top_features.parameters(), 'lr': 0.002},

        {"params": MODEL.cls_head.parameters(), 'lr': 0.0022},
        {"params": MODEL.base_classifier.parameters(), 'lr': 0.0025},
        {"params": MODEL.boxes_classifier.parameters(), 'lr': 0.0025},
        {"params": MODEL.final_classifier.parameters(), 'lr': 0.004},
    ]
)


LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=[4, 5, 6, 7, 8, 10, 11, 13, 14, 15], gamma=0.5),

]


EARLY_STOPPING_KWARGS = {
    'patience': 25,
    # 'score_function': None
}


LOG_INTERVAL = 100

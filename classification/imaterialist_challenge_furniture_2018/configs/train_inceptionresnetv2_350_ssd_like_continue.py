# Basic training configuration file
from pathlib import Path
import torch
from torch.optim import RMSprop
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, Compose
from torchvision.transforms import RandomResizedCrop, RandomAffine, RandomApply
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
# from models.inceptionresnetv2_ssd_like import FurnitureInceptionResNetV4350SSDLike


SEED = 17
DEBUG = True
DEVICE = 'cuda'

OUTPUT_PATH = "output"

size = 350

TRAIN_TRANSFORMS = Compose([
    RandomApply(
        [RandomAffine(degrees=30, translate=(0.2, 0.2), resample=3), ],
        p=0.5
    ),
    RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=3),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(hue=0.12, brightness=0.12),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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


# MODEL = FurnitureInceptionResNetV4350SSDLike(num_classes=128, pretrained='imagenet')
model_checkpoint = (Path(OUTPUT_PATH) / "train_inceptionresnetv2_350_ssd_like" / "20180512_1322" /
                    "model_FurnitureInceptionResNetV4350SSDLike_5_val_loss=0.6390978.pth").as_posix()
MODEL = torch.load(model_checkpoint)


N_EPOCHS = 100


OPTIM = RMSprop(
    params=[
        {"params": MODEL.extractor.stem.parameters(), 'lr': 0.00005},
        {"params": MODEL.extractor.low_features_a.parameters(), 'lr': 0.000145},
        {"params": MODEL.extractor.low_features_b.parameters(), 'lr': 0.000145},
        {"params": MODEL.extractor.mid_features.parameters(), 'lr': 0.00145},
        {"params": MODEL.extractor.top_features.parameters(), 'lr': 0.00145},
        {"params": MODEL.extractor.smooth_layers.parameters(), 'lr': 0.0145},
        {"params": MODEL.boxes_to_classes.parameters(), 'lr': 0.0145},
        {"params": MODEL.final_classifier.parameters(), 'lr': 0.0145},
    ],
    alpha=0.9,
    eps=1.0
)


LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=[2, 4, 6, 8, 10], gamma=0.2)
]


EARLY_STOPPING_KWARGS = {
    'patience': 25,
    # 'score_function': None
}


LOG_INTERVAL = 100

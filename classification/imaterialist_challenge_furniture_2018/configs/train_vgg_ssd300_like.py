# Basic training configuration file
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, Compose
from torchvision.transforms import Resize
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from models.vgg import FurnitureVGG16BNSSD300Like


SEED = 12
DEBUG = True
DEVICE = 'cuda'

OUTPUT_PATH = "output"


TRAIN_TRANSFORMS = Compose([
    Resize((300, 300), interpolation=3),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(hue=0.12, brightness=0.12),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


VAL_TRANSFORMS = TRAIN_TRANSFORMS


BATCH_SIZE = 120
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


MODEL = FurnitureVGG16BNSSD300Like(num_classes=128, pretrained=True)


N_EPOCHS = 100


OPTIM = Adam(
    params=[
        {"params": MODEL.extractor.top_features.parameters(), 'lr': 0.001},
        {"params": MODEL.boxes_to_classes.parameters(), 'lr': 0.001},
        {"params": MODEL.final_classifier.parameters(), 'lr': 0.0015},
    ],
)


LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=[2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20], gamma=0.92)
]


# REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=5, threshold=0.05, verbose=True)


EARLY_STOPPING_KWARGS = {
    'patience': 30,
    # 'score_function': None
}


LOG_INTERVAL = 100

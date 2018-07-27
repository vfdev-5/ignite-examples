# Basic training configuration file
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from ignite.engine import Events
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from common.sampling import SmartWeightedRandomSampler
from models.squeezenet_350 import FurnitureSqueezeNet350


SEED = 17
DEBUG = True

OUTPUT_PATH = "output"

size = 350

TRAIN_TRANSFORMS = [
    RandomResizedCrop(size, scale=(0.7, 1.0)),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(hue=0.1, brightness=0.2, contrast=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

VAL_TRANSFORMS = [
    RandomResizedCrop(size, scale=(0.8, 1.0)),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]


BATCH_SIZE = 180
NUM_WORKERS = 15


dataset = FilesFromCsvDataset("output/filtered_train_dataset.csv")
targets = [y - 1 for _, y in dataset]
train_sampler = SmartWeightedRandomSampler(targets)
TRAIN_LOADER = get_data_loader(dataset,
                               data_transform=TRAIN_TRANSFORMS,
                               batch_size=BATCH_SIZE,
                               sampler=train_sampler,
                               num_workers=NUM_WORKERS,
                               pin_memory=True)

val_dataset = FilesFromCsvDataset("output/filtered_val_dataset.csv")
VAL_LOADER = get_data_loader(val_dataset,
                             data_transform=VAL_TRANSFORMS,
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
    MultiStepLR(OPTIM, milestones=[4, 5, 7, 9, 10, 11, 12, 13], gamma=0.55)
]


EARLY_STOPPING_KWARGS = {
    'patience': 15,
    # 'score_function': None
}

LOG_INTERVAL = 100

TRAINER_CUSTOM_EVENT_HANDLERS = [
    # (event, handler), handler signature should be `foo(trainer, evaluator, logger)`

]


def smart_sampling_update(evaluator, trainer, logger):

    if trainer.state.epoch < 4:
        return

    assert hasattr(evaluator.state, "metrics"), "Evaluator state has no metrics"
    recall_per_class = evaluator.state.metrics['recall'].cpu()
    mean_recall_per_class = torch.mean(recall_per_class)
    low_recall_classes = np.where(recall_per_class < 0.6 * mean_recall_per_class)[0]
    logger.debug("Smart sampling update: low recall classes (< {}) : {}"
                 .format(mean_recall_per_class, low_recall_classes))
    n_classes = len(recall_per_class)
    class_weights = []
    for c in range(n_classes):
        w = 2.0 if c in low_recall_classes else 1.0
        class_weights.append((c, w))
    train_sampler.update_weights(class_weights)


EVALUATOR_CUSTOM_EVENT_HANDLERS = [
    # (event, handler), handler signature should be `foo(evaluator, trainer, logger)`
    (Events.COMPLETED, smart_sampling_update)
]

# Basic training configuration file
from pathlib import Path
import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import ColorJitter, ToTensor, Normalize
from ignite.engines import Events
from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader
from common.sampling import SmartWeightedRandomSampler
from models.inceptionv4 import FurnitureInceptionV4_350


SEED = 17
DEBUG = True

OUTPUT_PATH = "output"

size = 350

TRAIN_TRANSFORMS = [
    RandomResizedCrop(350, scale=(0.6, 1.0), interpolation=3),
    RandomVerticalFlip(p=0.5),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(hue=0.12, brightness=0.12),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

VAL_TRANSFORMS = [
    RandomResizedCrop(350, scale=(0.7, 1.0), interpolation=3),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]


BATCH_SIZE = 32
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


model_checkpoint = (Path(OUTPUT_PATH) / "train_inceptionv4_350_smart_sampler_resized_crop" /
                    "20180502_0902" /
                    "model_FurnitureInceptionV4_350_1_val_loss=0.7726298.pth").as_posix()
MODEL = torch.load(model_checkpoint)


N_EPOCHS = 100

OPTIM = SGD(
    params=[
        {"params": MODEL.stem.parameters(), 'lr': 0.0005},
        {"params": MODEL.features.parameters(), 'lr': 0.001},
        {"params": MODEL.classifier.parameters(), 'lr': 0.1},
    ],
    momentum=0.9)


LR_SCHEDULERS = [
    MultiStepLR(OPTIM, milestones=[2, 3, 4, 5, 6, 8, 10, 12], gamma=0.62)
]


REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=3, threshold=0.08, verbose=True)

EARLY_STOPPING_KWARGS = {
    'patience': 15,
    # 'score_function': None
}

LOG_INTERVAL = 100

TRAINER_CUSTOM_EVENT_HANDLERS = [
    # (event, handler), handler signature should be `foo(trainer, evaluator, logger)`

]


def smart_sampling_update(evaluator, trainer, logger):
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

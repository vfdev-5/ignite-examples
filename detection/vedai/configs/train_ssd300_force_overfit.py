# Training configuration
# Model: GeoSSD300

from pathlib import Path

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from torch.optim import SGD

from customized_torchcv.models.ssd import SSDBoxCoder, SSD300
from customized_torchcv.loss import SSDLoss

from common.dataset import VedaiFiles512x512, DatapointToImageBoxesLabels, TargetToCountedLabels
from common.transforms import TorchcvDetectionRandomTransform
from common.basic_data_loaders import get_trainval_data_loaders, get_trainval_indices


SEED = 12345
DEBUG = True

OUTPUT_PATH = "output"
DATASET_PATH = Path("/home/local_data/vedai/datasets/512x512")


TRAIN_TRANSFORMS = [
    DatapointToImageBoxesLabels(n_labels=11),
    TorchcvDetectionRandomTransform(mode='train', img_size=300, max_ratio=2),
]

VAL_TRANSFORMS = [
    DatapointToImageBoxesLabels(n_labels=11),
    TorchcvDetectionRandomTransform(mode='test', img_size=300, max_ratio=2),
]

BATCH_SIZE = 32
NUM_WORKERS = 10

MODEL = SSD300(num_classes=11)

box_coder = SSDBoxCoder(ssd_model=MODEL)
dataset = VedaiFiles512x512(path=DATASET_PATH, mode="train", fold_index=1, max_n_samples=200)
trainval_split = MultilabelStratifiedKFold(n_splits=7, shuffle=True, random_state=SEED)
train_index, val_index = get_trainval_indices(dataset,
                                              trainval_split,
                                              fold_index=0,
                                              xy_transforms=[TargetToCountedLabels(n_labels=11), ],
                                              batch_size=BATCH_SIZE, n_workers=NUM_WORKERS)

TRAIN_LOADER, VAL_LOADER = get_trainval_data_loaders(dataset,
                                                     train_index, val_index,
                                                     train_transforms=TRAIN_TRANSFORMS,
                                                     val_transforms=VAL_TRANSFORMS,
                                                     box_coder=box_coder,
                                                     train_batch_size=BATCH_SIZE,
                                                     val_batch_size=BATCH_SIZE,
                                                     num_workers=NUM_WORKERS, pin_memory=True)

N_EPOCHS = 100


import torch.nn.functional as F


class _XEntropy(SSDLoss):

    def forward(self, y_preds, y_true):
        loc_preds, cls_preds = y_preds
        loc_targets, cls_targets = y_true
        pos = cls_targets > 0  # [N,#anchors]
        batch_size = pos.size(0)
        num_pos = pos.data.long().sum()

        #===============================================================
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        #===============================================================
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], size_average=False)

        return loc_loss / (num_pos + 1e-8)


CRITERION = _XEntropy(num_classes=MODEL.num_classes)

OPTIM = SGD(
    params=[
        {"params": MODEL.parameters(), 'lr': 0.01},
        # {"params": MODEL.classifier.parameters(), 'lr': 0.1},
    ])

LOG_INTERVAL = 100



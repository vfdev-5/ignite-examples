# Training configuration 
# Model: GeoSSD300

from pathlib import Path

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from customized_torchcv.loss import SSDLoss

from common.dataset import VedaiFiles512x512, DatapointToImageBoxesLabels, TargetToCountedLabels
from common.transforms import TorchcvDetectionRandomTransform
from common.basic_data_loaders import get_trainval_data_loaders, get_trainval_resampled_indices

from models.net import AnotherSSD300, BoxCoder300


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

MODEL = AnotherSSD300(n_classes=11)

box_coder = BoxCoder300(feature_map_sizes=MODEL.feature_map_sizes)
dataset = VedaiFiles512x512(path=DATASET_PATH, mode="train", fold_index=1)
trainval_split = MultilabelStratifiedKFold(n_splits=7, shuffle=True, random_state=SEED)
train_index, val_index = get_trainval_resampled_indices(dataset,
                                                        trainval_split,
                                                        fold_index=0,
                                                        xy_transforms=[TargetToCountedLabels(n_labels=11), ],
                                                        batch_size=BATCH_SIZE, n_workers=NUM_WORKERS,
                                                        seed=SEED)

TRAIN_LOADER, VAL_LOADER = get_trainval_data_loaders(dataset,
                                                     train_index, val_index,
                                                     train_transforms=TRAIN_TRANSFORMS,
                                                     val_transforms=VAL_TRANSFORMS,
                                                     box_coder=box_coder,
                                                     train_batch_size=BATCH_SIZE,
                                                     val_batch_size=BATCH_SIZE,
                                                     num_workers=NUM_WORKERS, pin_memory=True)

N_EPOCHS = 100


CRITERION = SSDLoss(num_classes=MODEL.n_classes)

OPTIM = SGD(
    params=[
        {"params": MODEL.parameters(), 'lr': 0.001},
        # {"params": MODEL.classifier.parameters(), 'lr': 0.1},
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



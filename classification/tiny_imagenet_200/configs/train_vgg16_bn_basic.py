# Basic training configuration file
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, RandomCrop
from torchvision.transforms import ColorJitter, ToTensor
from models.small_vgg16_bn import get_small_vgg16_bn
from transforms import GlobalContrastNormalize


SEED = 12345
DEBUG = True

OUTPUT_PATH = "/home/project/tiny_imagenet200_output"
DATASET_PATH = "/home/local_data/tiny-imagenet-200/"
TRAINVAL_SPLIT = {
    'fold_index': 0,
    'n_splits': 5
}

MODEL = get_small_vgg16_bn(200)

N_EPOCHS = 100

BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
NUM_WORKERS = 8

OPTIM = SGD(MODEL.parameters(), lr=0.1)

# LR_SCHEDULERS = [
#     MultiStepLR(OPTIM, milestones=[55, 70, 80, 90, 100], gamma=0.5)
# ]

TRAIN_TRANSFORMS = [
    RandomCrop(size=64, padding=10),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ColorJitter(hue=0.1, brightness=0.1),
    ToTensor(),
    # https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/datasets/make_cifar10_gcn_whitened.py#L19
    GlobalContrastNormalize(scale=55.0)
]
VAL_TRANSFORMS = [
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ToTensor(),
    # https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/datasets/make_cifar10_gcn_whitened.py#L19
    GlobalContrastNormalize(scale=55.0)
]

REDUCE_LR_ON_PLATEAU = ReduceLROnPlateau(OPTIM, mode='min', factor=0.5, patience=5, threshold=0.05, verbose=True)

EARLY_STOPPING_KWARGS = {
    'patience': 30,
    # 'score_function': None
}

LOG_INTERVAL = 100

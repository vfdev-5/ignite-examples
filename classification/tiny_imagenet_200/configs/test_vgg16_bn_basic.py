# Basic evaluation configuration file
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip
from torchvision.transforms import ToTensor
from transforms import GlobalContrastNormalize


SEED = 12345
DEBUG = True

OUTPUT_PATH = "output"
DATASET_PATH = "/home/local_data/tiny-imagenet-200/"

MODEL = OUTPUT_PATH + "/training_VGG_20180503_1450/model_VGG_2_val_loss=4.655257.pth"

N_TTA = 10

BATCH_SIZE = 128
NUM_WORKERS = 8

TEST_TRANSFORMS = [
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ToTensor(),
    # https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/datasets/make_cifar10_gcn_whitened.py#L19
    GlobalContrastNormalize(scale=55.0)
]

LOG_INTERVAL = 100

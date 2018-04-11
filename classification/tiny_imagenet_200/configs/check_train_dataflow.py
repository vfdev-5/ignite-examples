# Check training dataflow
from configs.train_vgg16_bn_basic import TRAIN_TRANSFORMS, VAL_TRANSFORMS, TRAINVAL_SPLIT
from dataflow import get_trainval_data_loaders

SEED = 12345
DEBUG = True

OUTPUT_PATH = "/home/working_directory/ml/tmp/tiny_imagenet_200_output/train_dataflow"
DATASET_PATH = "/home/fast_storage/tiny-imagenet-200"

N_EPOCHS = 1

BATCH_SIZE = 128
NUM_WORKERS = 8

DATA_LOADER, _ = get_trainval_data_loaders(DATASET_PATH,
                                           TRAIN_TRANSFORMS,
                                           VAL_TRANSFORMS,
                                           BATCH_SIZE, BATCH_SIZE,
                                           TRAINVAL_SPLIT,
                                           NUM_WORKERS, seed=SEED, cuda=True)
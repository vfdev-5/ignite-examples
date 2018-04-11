# Check training dataflow
from configs.train_vgg16_bn_basic import DATASET_PATH, VAL_LOADER

SEED = 12345
DEBUG = True

OUTPUT_PATH = "output/val_dataflow"

N_CLASSES = 128
N_EPOCHS = 1
DATA_LOADER = VAL_LOADER
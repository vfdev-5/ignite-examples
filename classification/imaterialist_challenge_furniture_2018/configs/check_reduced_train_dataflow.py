# Check training dataflow
from configs.train_squeezenet_350_on_reduced_dataset import TRAIN_LOADER

SEED = 12345
DEBUG = True

OUTPUT_PATH = "output/reduced_train_dataflow"

N_CLASSES = 128
N_EPOCHS = 1
DATA_LOADER = TRAIN_LOADER
# Check training dataflow
from configs.train_inceptionresnetv2_350_weighted_sampler_resized_crop import TRAIN_LOADER

SEED = 12345
DEBUG = True

OUTPUT_PATH = "output/train_weighted_dataflow"

N_CLASSES = 128
DATA_LOADER = TRAIN_LOADER

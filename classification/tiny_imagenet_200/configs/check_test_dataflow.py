# Check test dataflow
from configs.test_vgg16_bn_basic import TEST_TRANSFORMS
from dataflow import get_test_data_loader

SEED = 12345
DEBUG = True

OUTPUT_PATH = "output/test_dataflow"
DATASET_PATH = "/home/fast_storage/tiny-imagenet-200"

N_EPOCHS = 1

BATCH_SIZE = 128
NUM_WORKERS = 8

DATA_LOADER = get_test_data_loader(DATASET_PATH,
                                   TEST_TRANSFORMS,
                                   BATCH_SIZE,
                                   NUM_WORKERS, device='cuda')

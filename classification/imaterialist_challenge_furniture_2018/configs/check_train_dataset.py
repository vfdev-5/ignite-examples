# Check training dataflow
from pathlib import Path
from torchvision.transforms import ToTensor
from common.data_loaders import get_data_loader


SEED = 12345
DEBUG = True

OUTPUT_PATH = "output/train_dataflow"
DATASET_PATH = Path("/home/fast_storage/imaterialist-challenge-furniture-2018/")


N_CLASSES = 128
N_EPOCHS = 1


BATCH_SIZE = 1
NUM_WORKERS = 8

DATA_LOADER = get_data_loader(
    dataset_path=DATASET_PATH / "train",
    data_transform=[ToTensor(), ],
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    cuda=False)

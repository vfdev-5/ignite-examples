# Basic training configuration file
from pathlib import Path
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import ToTensor, Normalize, ColorJitter
from common.dataset import get_test_data_loader


SEED = 17
DEBUG = True

OUTPUT_PATH = "output"
dataset_path = Path("/home/local_data/imaterialist-challenge-furniture-2018/")

SAVE_PROBAS = True
# SAMPLE_SUBMISSION_PATH = dataset_path / "sample_submission_randomlabel.csv"


TEST_TRANSFORMS = [
    RandomResizedCrop(350, scale=(0.7, 1.0), interpolation=3),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(hue=0.12, brightness=0.12),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

N_CLASSES = 128
BATCH_SIZE = 64
NUM_WORKERS = 15

TEST_LOADER = get_test_data_loader(
    dataset_path=dataset_path / "test",
    test_data_transform=TEST_TRANSFORMS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True)


MODEL = (Path(OUTPUT_PATH) / "train_nasnetalarge_350_random_resized_crop" / "20180509_1544" /
         "model_FurnitureNASNetALarge350_5_val_loss=0.5331665.pth").as_posix()

N_TTA = 12

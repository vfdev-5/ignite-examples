# Basic training configuration file
from pathlib import Path
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, CenterCrop
from torchvision.transforms import RandomApply, RandomAffine
from torchvision.transforms import ToTensor, Normalize
from common.dataset import get_test_data_loader


SEED = 12345
DEBUG = True

OUTPUT_PATH = "output"
dataset_path = Path("/home/fast_storage/imaterialist-challenge-furniture-2018/")
SAMPLE_SUBMISSION_PATH = dataset_path / "sample_submission_randomlabel.csv"


TEST_TRANSFORMS = [
    RandomApply(
        [RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.7, 1.2), resample=2), ],
        p=0.5
    ),
    CenterCrop(size=350),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

N_CLASSES = 128
BATCH_SIZE = 24
NUM_WORKERS = 8

TEST_LOADER = get_test_data_loader(
    dataset_path=dataset_path / "test_400x400",
    test_data_transform=TEST_TRANSFORMS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True)


MODEL = (Path(OUTPUT_PATH) / "training_FurnitureInceptionV4_350_20180418_1903" /
         "model_FurnitureInceptionV4_350_6_val_loss=0.5831019.pth").as_posix()

N_TTA = 10

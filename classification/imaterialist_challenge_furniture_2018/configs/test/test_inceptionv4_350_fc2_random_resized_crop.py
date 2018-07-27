# Basic training configuration file
from pathlib import Path
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import ToTensor, Normalize
from common.dataset import get_test_data_loader


SEED = 42
DEBUG = True

OUTPUT_PATH = Path("output") / "test"
dataset_path = Path("/home/fast_storage/imaterialist-challenge-furniture-2018/")
SAMPLE_SUBMISSION_PATH = dataset_path / "sample_submission_randomlabel.csv"


TEST_TRANSFORMS = [
    RandomResizedCrop(350, scale=(0.7, 1.0), interpolation=3),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

N_CLASSES = 128
BATCH_SIZE = 128
NUM_WORKERS = 15

TEST_LOADER = get_test_data_loader(
    dataset_path=dataset_path / "test",
    test_data_transform=TEST_TRANSFORMS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True)


MODEL = (Path("output") / "train" / "train_inceptionv4_350_fc2_random_resized_crop" / "20180515_2213" /
         "model_FurnitureInceptionV4_350_FC2_11_val_loss=0.6425431.pth").as_posix()

N_TTA = 12

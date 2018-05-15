# Basic training configuration file
from pathlib import Path
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, CenterCrop
from torchvision.transforms import RandomApply, RandomAffine
from torchvision.transforms import ToTensor, Normalize
from common.dataset import get_test_data_loader


SEED = 12345
DEBUG = True

OUTPUT_PATH = "output"
dataset_path = Path("/home/storage_ext4_1tb/imaterialist-challenge-furniture-2018/")

SAVE_PROBAS = True
# SAMPLE_SUBMISSION_PATH = dataset_path / "sample_submission_randomlabel.csv"


TEST_TRANSFORMS = [
    RandomApply(
        [RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.7, 1.2), resample=2), ],
        p=0.5
    ),
    CenterCrop(size=350),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

N_CLASSES = 128
BATCH_SIZE = 32
NUM_WORKERS = 8

TEST_LOADER = get_test_data_loader(
    dataset_path=dataset_path / "val_400x400",
    test_data_transform=TEST_TRANSFORMS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True)


MODEL = (Path(OUTPUT_PATH) / "training_FurnitureSqueezeNet350_20180414_1610" /
         "model_FurnitureSqueezeNet350_47_val_loss=0.8795085.pth").as_posix()

N_TTA = 1



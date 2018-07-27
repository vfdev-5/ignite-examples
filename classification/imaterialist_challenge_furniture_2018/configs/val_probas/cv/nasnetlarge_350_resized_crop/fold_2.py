# Basic training configuration file
from pathlib import Path
from torch.utils.data import ConcatDataset
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import ToTensor, Normalize

from common.dataset import FilesFromCsvDataset
from common.data_loaders import get_data_loader, get_trainval_indices


SEED = 1245
DEBUG = True
DEVICE = "cuda"

OUTPUT_PATH = Path("output") / "val_probas" / "cv" / "nasnetlarge_350_resized_crop"
dataset_path = Path("/home/fast_storage/imaterialist-challenge-furniture-2018/")

SAVE_PROBAS = True
# SAMPLE_SUBMISSION_PATH = dataset_path / "sample_submission_randomlabel.csv"


TEST_TRANSFORMS = [
    RandomResizedCrop(350, scale=(0.7, 1.0), interpolation=3),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

N_CLASSES = 128
batch_size = 64
num_workers = 15

train_dataset = FilesFromCsvDataset("output/unique_filtered_train_dataset.csv")
val_dataset = FilesFromCsvDataset("output/unique_filtered_val_dataset.csv")
trainval_dataset = ConcatDataset([train_dataset, val_dataset])


# #### Stratified split :
fold_index = 2
n_splits = 4
train_index, val_index = get_trainval_indices(trainval_dataset,
                                              fold_index=fold_index, n_splits=n_splits,
                                              xy_transforms=None,
                                              batch_size=batch_size, n_workers=8,
                                              seed=SEED)
# ####

TEST_LOADER = get_data_loader(trainval_dataset,
                              data_transform=TEST_TRANSFORMS,
                              sample_indices=val_index,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory="cuda" in DEVICE)


MODEL = (Path("output") / "cv" / "nasnetlarge_350_resized_crop" / "fold_2" / "20180525_1653" /
         "model_FurnitureNASNetALarge350_9_val_loss=0.4992649.pth").as_posix()

N_TTA = 7

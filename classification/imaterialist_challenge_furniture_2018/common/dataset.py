from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import Compose


class TrainvalFilesDataset(Dataset):

    def __init__(self, path, corrupted_files=None):
        self.path = Path(path)
        assert self.path.exists(), "Train/Val dataset is not found at '{}'".format(path)
        files = self.path.glob("*.png")
        self.images = [f.as_posix() for f in files]
        # remove corrupted files:
        if corrupted_files is not None:
            for f in corrupted_files:
                self.images.remove(f)
        self.n = len(self.images)
        self.labels = [None] * self.n
        for i, f in enumerate(self.images):
            self.labels[i] = int(Path(f).stem.split('_')[1])
        self.unique_labels = np.unique(self.labels)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class FilesFromCsvDataset(Dataset):

    def __init__(self, csv_filepath):
        self.csv_filepath = Path(csv_filepath)
        assert self.csv_filepath.exists(), "CSV filepath '{}' is not found".format(csv_filepath)
        df = pd.read_csv(self.csv_filepath)
        self.n = len(df)
        self.images = df['filepath'].values
        self.labels = df['label'].values

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class TestFilesDataset(Dataset):

    def __init__(self, path):
        path = Path(path)
        assert path.exists(), "Test dataset is not found at '{}'".format(path)
        files = path.glob("*.png")
        self.images = [f for f in files]
        if "_" in self.images[0].stem:
            self.image_ids = [self.train_filepath_to_image_id(f) for f in self.images]
        else:
            self.image_ids = [self.test_filepath_to_image_id(f) for f in self.images]
        self.n = len(self.images)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.images[index].as_posix(), self.image_ids[index]

    @staticmethod
    def train_filepath_to_image_id(filepath):
        stem = filepath.stem
        split = stem.split("_")
        return int(split[0])

    @staticmethod
    def test_filepath_to_image_id(filepath):
        stem = filepath.stem
        return int(stem)


def read_image(fp):
    return Image.open(fp)


class TransformedDataset(Dataset):

    def __init__(self, dataset, transforms, target_transforms=None):
        assert isinstance(dataset, Dataset)
        assert callable(transforms)
        if target_transforms is not None:
            assert callable(target_transforms)

        self.dataset = dataset
        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = self.transforms(img)
        if self.target_transforms is not None:
            label = self.target_transforms(label)
        return img, label


def get_data_loaders(train_dataset_path,
                     val_dataset_path,
                     train_data_transform,
                     val_data_transform,
                     train_batch_size, val_batch_size,
                     num_workers,
                     pin_memory=True,
                     collate_fn=default_collate):
    if isinstance(train_data_transform, (list, tuple)):
        train_data_transform = Compose(train_data_transform)

    if isinstance(val_data_transform, (list, tuple)):
        val_data_transform = Compose(val_data_transform)

    train_dataset = TrainvalFilesDataset(train_dataset_path)
    val_dataset = TrainvalFilesDataset(val_dataset_path)

    train_dataset = TransformedDataset(train_dataset, transforms=read_image,
                                       target_transforms=lambda l: l - 1)
    val_dataset = TransformedDataset(val_dataset, transforms=read_image,
                                     target_transforms=lambda l: l - 1)
    train_dataset = TransformedDataset(train_dataset, transforms=train_data_transform)
    val_dataset = TransformedDataset(val_dataset, transforms=val_data_transform)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True,
                            collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


def get_test_data_loader(dataset_path,
                         test_data_transform,
                         batch_size,
                         num_workers, pin_memory=True):

    if isinstance(test_data_transform, (list, tuple)):
        test_data_transform.insert(0, read_image)
        test_data_transform = Compose(test_data_transform)

    test_dataset = TestFilesDataset(dataset_path)
    test_dataset = TransformedDataset(test_dataset, transforms=test_data_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return test_loader


def get_train_eval_data_loader(train_loader, indices=None):

    assert isinstance(indices, (list, tuple, np.ndarray))
    subset = Subset(train_loader.dataset, indices)
    train_eval_loader = DataLoader(subset, batch_size=train_loader.batch_size,
                                   shuffle=False, drop_last=False,
                                   num_workers=train_loader.num_workers,
                                   pin_memory=train_loader.pin_memory,
                                   collate_fn=train_loader.collate_fn,
                                   timeout=train_loader.timeout,
                                   worker_init_fn=train_loader.worker_init_fn)
    return train_eval_loader
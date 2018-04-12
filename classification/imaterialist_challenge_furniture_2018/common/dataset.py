from pathlib import Path

import numpy as np

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
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


class TestFilesDataset(Dataset):

    def __init__(self, path):
        path = Path(path)
        assert path.exists(), "Test dataset is not found at '{}'".format(path)
        files = path.glob("*.png")
        self.images = [f for f in files]
        self.n = len(self.images)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.images[index].as_posix(), int(self.images[index].stem)


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
                     num_workers, cuda=True):
    if isinstance(train_data_transform, (list, tuple)):
        train_data_transform = Compose(train_data_transform)

    if isinstance(val_data_transform, (list, tuple)):
        val_data_transform = Compose(val_data_transform)

    corrupted_files = [
        # (Path(train_dataset_path) / "124682_20.png").as_posix(),
    ]
    train_dataset = TrainvalFilesDataset(train_dataset_path, corrupted_files=corrupted_files)
    val_dataset = TrainvalFilesDataset(val_dataset_path)

    train_dataset = TransformedDataset(train_dataset, transforms=read_image,
                                       target_transforms=lambda l: l - 1)
    val_dataset = TransformedDataset(val_dataset, transforms=read_image,
                                     target_transforms=lambda l: l - 1)
    train_dataset = TransformedDataset(train_dataset, transforms=train_data_transform)
    val_dataset = TransformedDataset(val_dataset, transforms=val_data_transform)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              shuffle=True,
                              # sampler=SubsetRandomSampler(train_indices),
                              num_workers=num_workers, pin_memory=cuda)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=cuda)

    return train_loader, val_loader


def get_test_data_loader(dataset_path,
                         test_data_transform,
                         batch_size,
                         num_workers, cuda=True):

    if isinstance(test_data_transform, (list, tuple)):
        test_data_transform.insert(0, read_image)
        test_data_transform = Compose(test_data_transform)

    test_dataset = TestFilesDataset(dataset_path)
    test_dataset = TransformedDataset(test_dataset, transforms=test_data_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False,
                             num_workers=num_workers, pin_memory=cuda)
    return test_loader

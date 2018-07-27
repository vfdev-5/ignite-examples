import os

import numpy as np

from sklearn.model_selection import StratifiedKFold
from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset
from torchvision.datasets.folder import find_classes, has_file_allowed_extension, IMG_EXTENSIONS, pil_loader


def get_train_val_indices(data_loader, fold_index=0, n_splits=5, seed=None):
    # Stratified split: train/val:
    n_samples = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    X = np.zeros((n_samples, 1))
    y = np.zeros(n_samples, dtype=np.int)
    for i, (_, label) in enumerate(data_loader):
        start_index = batch_size * i
        end_index = batch_size * (i + 1)
        y[start_index: end_index] = label.numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_indices, val_indices) in enumerate(skf.split(X, y)):
        if i == fold_index:
            return train_indices, val_indices


def get_trainval_data_loaders(dataset_path,
                              train_data_transform,
                              val_data_transform,
                              train_batch_size, val_batch_size,
                              trainval_split,
                              num_workers, seed=None, device='cpu'):
    if isinstance(train_data_transform, (list, tuple)):
        train_data_transform = Compose(train_data_transform)

    if isinstance(val_data_transform, (list, tuple)):
        val_data_transform = Compose(val_data_transform)

    train_dataset_path = os.path.join(dataset_path, 'train')
    train_dataset = ImageFolder(train_dataset_path, transform=train_data_transform)
    val_dataset = ImageFolder(train_dataset_path, transform=val_data_transform)

    # Temporary dataset and dataloader to create train/val split
    trainval_dataset = ImageFolder(train_dataset_path, transform=ToTensor())
    trainval_loader = DataLoader(trainval_dataset, batch_size=train_batch_size,
                                 shuffle=False, drop_last=False,
                                 num_workers=num_workers, pin_memory=False)

    train_indices, val_indices = get_train_val_indices(trainval_loader, seed=seed, **trainval_split)

    pin_memory = 'cuda' in device
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              sampler=SubsetRandomSampler(train_indices),
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size,
                            sampler=SubsetRandomSampler(val_indices),
                            num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


class TestDataset(Dataset):

    def __init__(self, root, loader=None, transform=None):
        assert os.path.exists(root)
        self.classes, class_to_idx = find_classes(os.path.join(root, 'train'))
        self.image_paths = []
        path = os.path.join(root, 'test', 'images')
        for p, _, fnames in sorted(os.walk(path)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                    path = os.path.join(p, fname)
                    self.image_paths.append(path)
        self.loader = pil_loader if loader is None else loader
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path


def get_test_data_loader(dataset_path,
                         test_data_transform,
                         batch_size,
                         num_workers, device='cpu'):

    if isinstance(test_data_transform, (list, tuple)):
        test_data_transform = Compose(test_data_transform)

    test_dataset = TestDataset(dataset_path, transform=test_data_transform)

    pin_memory = 'cuda' in device
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

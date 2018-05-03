from pathlib import Path
import numpy as np

from sklearn.model_selection import StratifiedKFold

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose

from common.dataset import TransformedDataset, read_image, TrainvalFilesDataset


def get_trainval_data_loaders(dataset, train_index, val_index,
                              train_transforms=None,
                              val_transforms=None,
                              train_batch_size=32,
                              val_batch_size=32,
                              collate_fn=default_collate,
                              num_workers=8,
                              pin_memory=True):
    assert isinstance(dataset, Dataset)
    if train_transforms is not None:
        assert isinstance(train_transforms, (list, tuple))
        train_transforms = Compose(train_transforms)

    if val_transforms is not None:
        assert isinstance(val_transforms, (list, tuple))
        val_transforms = Compose(val_transforms)

    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)

    dataset = TransformedDataset(dataset, transforms=read_image,
                                 target_transforms=lambda l: l - 1)

    train_aug_dataset = TransformedDataset(dataset, transforms=train_transforms)
    val_aug_dataset = TransformedDataset(dataset, transforms=val_transforms)

    train_batches = DataLoader(train_aug_dataset, batch_size=train_batch_size,
                               sampler=train_sampler,
                               num_workers=num_workers,
                               collate_fn=collate_fn,
                               pin_memory=True, drop_last=True)

    val_batches = DataLoader(val_aug_dataset, batch_size=val_batch_size,
                             sampler=val_sampler,
                             num_workers=num_workers,
                             collate_fn=collate_fn,
                             pin_memory=pin_memory, drop_last=True)

    return train_batches, val_batches


def get_trainval_indices(dataset, fold_index=0, n_splits=5, xy_transforms=None, batch_size=32, n_workers=8):

    trainval_split = StratifiedKFold(n_splits=n_splits, shuffle=True)

    if xy_transforms is not None:
        targets_dataset = TransformedDataset(dataset, transforms=xy_transforms)
    else:
        targets_dataset = dataset

    n_samples = len(targets_dataset)
    x = np.zeros((n_samples, 1))
    y = np.zeros((n_samples,), dtype=np.uint8)

    def id_collate_fn(_x):
        return _x

    data_loader = DataLoader(targets_dataset, batch_size=batch_size, num_workers=n_workers, collate_fn=id_collate_fn)

    for i, dp in enumerate(data_loader):
        y[i * batch_size: (i + 1) * batch_size] = [p[1] for p in dp]

    train_index = None
    test_index = None
    for i, (train_index, test_index) in enumerate(trainval_split.split(x, y)):
        if i == fold_index:
            break
    return train_index, test_index


def get_data_loader(dataset_or_path,
                    data_transform=None,
                    sample_indices=None,
                    sampler=None,
                    collate_fn=default_collate,
                    batch_size=16,
                    num_workers=8, cuda=True):
    assert isinstance(dataset_or_path, Dataset) or \
        (isinstance(dataset_or_path, (str, Path)) and Path(dataset_or_path).exists()), \
        "Dataset or path should be either Dataset or path to images, but given {}".format(dataset_or_path)

    assert sample_indices is None or sampler is None, "Both are not possible"

    if data_transform is not None and isinstance(data_transform, (list, tuple)):
        data_transform = Compose(data_transform)

    if isinstance(dataset_or_path, (str, Path)) and Path(dataset_or_path).exists():
        dataset = TrainvalFilesDataset(dataset_or_path)
    else:
        dataset = dataset_or_path

    if sample_indices is None and sampler is None:
        sample_indices = np.arange(len(dataset))

    if sample_indices is not None:
        sampler = SubsetRandomSampler(sample_indices)

    dataset = TransformedDataset(dataset, transforms=read_image, target_transforms=lambda l: l - 1)
    if data_transform is not None:
        dataset = TransformedDataset(dataset, transforms=data_transform)

    data_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=sampler,
                             collate_fn=collate_fn,
                             num_workers=num_workers, pin_memory=cuda)
    return data_loader

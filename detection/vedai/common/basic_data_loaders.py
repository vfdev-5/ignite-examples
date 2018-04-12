import numpy as np

from sklearn.model_selection._split import _BaseKFold

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from common.dataset import TransformedDataset


def get_trainval_data_loaders(dataset,
                              train_index, val_index,
                              train_transforms, val_transforms,
                              box_coder=None,
                              train_batch_size=32,
                              val_batch_size=32,
                              num_workers=8, pin_memory=True):
    assert isinstance(dataset, Dataset)
    assert isinstance(train_transforms, (list, tuple))
    assert isinstance(val_transforms, (list, tuple))

    if box_coder is not None:
        def box_encode(img, boxes_labels):
            boxes, labels = boxes_labels
            return img, box_coder.encode(boxes, labels)

        train_transforms.append(box_encode)
        val_transforms.append(box_encode)

    train_sampler = SubsetRandomSampler(train_index)
    val_sampler = SubsetRandomSampler(val_index)

    train_aug_dataset = TransformedDataset(dataset, xy_transforms=train_transforms)
    val_aug_dataset = TransformedDataset(dataset, xy_transforms=val_transforms, return_input=True)

    train_batches = DataLoader(train_aug_dataset, batch_size=train_batch_size,
                               sampler=train_sampler,
                               num_workers=num_workers,
                               pin_memory=True, drop_last=True)

    val_batches = DataLoader(val_aug_dataset, batch_size=val_batch_size,
                             sampler=val_sampler,
                             num_workers=num_workers,
                             pin_memory=pin_memory, drop_last=True)

    return train_batches, val_batches


def get_trainval_resampled_indices(dataset, trainval_split, fold_index=0, xy_transforms=None,
                                   batch_size=32, n_workers=8, seed=None):

    train_index, val_index, y = get_trainval_indices(dataset, trainval_split,
                                                     fold_index=fold_index,
                                                     xy_transforms=xy_transforms,
                                                     batch_size=batch_size,
                                                     n_workers=n_workers, return_targets=True)

    y_unique = np.unique(y, axis=0)
    y_ohe_to_index_map = dict([(tuple(y.tolist()), i) for i, y in enumerate(y_unique)])

    y_indices = np.zeros((len(y),), dtype=np.uint8)
    for i, v in enumerate(y):
        y_indices[i] = y_ohe_to_index_map[tuple(v)]

    train_index_resampled, _ = resample_indices(train_index, y_indices[train_index], seed=seed)
    val_index_resampled, _ = resample_indices(val_index, y_indices[val_index], seed=seed)
    return train_index_resampled, val_index_resampled


def get_trainval_indices(dataset, trainval_split, fold_index=0, xy_transforms=None,
                         batch_size=32, n_workers=8, return_targets=False):

    assert isinstance(trainval_split, _BaseKFold)
    targets_dataset = TransformedDataset(dataset, xy_transforms=xy_transforms)

    n_samples = len(targets_dataset)
    n_labels = len(targets_dataset[0][1])
    x = np.zeros((n_samples, 1))
    y = np.zeros((n_samples, n_labels), dtype=np.uint8)

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
    if return_targets:
        return train_index, test_index, y
    return train_index, test_index


def resample_indices(indices, y, seed=None):
    # Random Under-sample the first class:
    y_stats = np.bincount(y)
    n_samples = int(np.mean(y_stats[1:]))
    desired_n_samples_per_clases = {0: n_samples}
    for i, v in enumerate(y_stats[1:]):
        if v > 0:
            desired_n_samples_per_clases[i + 1] = v

    rus = RandomUnderSampler(random_state=seed, ratio=desired_n_samples_per_clases)
    indices_resampled, y_resampled = rus.fit_sample(indices[:, None], y)

    # Random Over-sample all classes
    ros = RandomOverSampler(random_state=seed)
    indices_resampled, y_resampled = ros.fit_sample(indices_resampled, y_resampled)

    return indices_resampled[:, 0], y_resampled






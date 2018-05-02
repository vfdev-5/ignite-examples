import warnings

import numpy as np

import torch
from torch.utils.data.sampler import WeightedRandomSampler

from imblearn.under_sampling import RandomUnderSampler


def get_reduced_train_indices(dataset, max_n_samples_per_class, seed=None):

    n_samples = len(dataset)
    indices = np.zeros((n_samples, 1), dtype=np.int)
    y = np.zeros((n_samples,), dtype=np.int)

    for i, dp in enumerate(dataset):
        y[i] = dp[1]
        indices[i, 0] = i

    y_counts = np.bincount(y)
    desired_ratio = dict([
        (label, min(c, max_n_samples_per_class)) for label, c in enumerate(y_counts)
    ])
    rs = RandomUnderSampler(ratio=desired_ratio, random_state=seed)
    resampled_indices, y_resampled = rs.fit_sample(indices, y)
    return resampled_indices[:, 0]


def get_weighted_train_sampler(dataset, classes_weight, n_samples=25000):

    assert isinstance(classes_weight, np.ndarray) and classes_weight.ndim == 2
    weights = np.ones((len(dataset),))

    y = np.zeros((len(dataset, )), dtype=np.int)
    for i, dp in enumerate(dataset):
        y[i] = dp[1] - 1

    for c, w in classes_weight:
        indices = np.where(y == int(c))[0]
        weights[indices] = w
    sampler = WeightedRandomSampler(weights, num_samples=n_samples)
    return sampler


class SmartWeightedRandomSampler(WeightedRandomSampler):

    def __init__(self, targets, num_samples=None):
        weights = np.ones((len(targets),))

        if num_samples is None:
            num_samples = len(targets)

        self.weights_indices_per_class = {}
        unique_classes = np.unique(targets)
        assert len(unique_classes) > 0
        for c in unique_classes:
            indices = np.where(targets == c)[0].tolist()
            assert len(indices) > 0, "No targets found for the class {}".format(c)
            self.weights_indices_per_class[c] = indices
        assert len(self.weights_indices_per_class) == len(unique_classes)

        super(SmartWeightedRandomSampler, self).__init__(weights, num_samples, replacement=True)

    def update_weights(self, class_weights):
        for c, w in class_weights:
            if c not in self.weights_indices_per_class:
                warnings.warn("Class {} is not found in classes with weight indices: {}"
                              .format(c, list(self.weights_indices_per_class.keys())))
                continue
            indices = self.weights_indices_per_class[c]
            self.weights[indices] = w

    def reset_weights(self):
        self.weights = torch.ones_like(self.weights)

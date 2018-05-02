import numpy as np

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

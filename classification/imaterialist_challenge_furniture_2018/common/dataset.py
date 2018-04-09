from pathlib import Path

import numpy as np

from torch.utils.data import Dataset


class TrainvalFilesDataset(Dataset):

    def __init__(self, path):
        path = Path(path)
        assert path.exists(), "Train/Val dataset is not found at '{}'".format(path)
        files = path.glob("*.png")
        self.images = [f.as_posix() for f in files]
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
        self.images = [f.as_posix() for f in files]
        self.n = len(self.images)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.images[index]

import random
from pathlib import Path

import numpy as np

from torch.utils.data import Dataset

from PIL import Image


def imread_pillow(filename):
    try:
        return Image.open(filename)
    except Exception as e:
        raise RuntimeError("Failed to read image at '{}'".format(filename) +
                           "Error: {}".format(e))


class _VedaiFiles:
    
    def __init__(self, path, img_type='co', option='512'):
        path = Path(path)
        assert img_type in ('co', 'ir')
        assert option in ('512', '1024')
        assert path.exists()
        self.path = path
        self.img_type = img_type
        targets = [t for t in (path / "Annotations{}".format(option)).glob("0*.txt")]
        self.targets = dict([(t.stem, t.as_posix()) for t in targets])
        self.images = dict([(t.stem, (path / "Vehicules{}".format(option) /
                                      (t.stem + "_{}.png".format(self.img_type))).as_posix()) for t in targets])
        for fp in self.images.values():
            assert Path(fp).exists(), "{} is not found".format(fp)
        self.ids = sorted(self.targets)
        
        self.n_folds = 10
        self.classes = {
            1: "car",
            2: "truck",            
            4: "tractor",
            5: "camping car",
            23: "boat",
            7: "x",
            8: "xx",            
            9: "van",
            10: "other",
            11: "pickup",
            31: "large"
        }
                                
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        return self.images[index], self.targets[index]
        
    def _get_fold_indices(self, fold_index, mode):        
        fp = self.path / "Annotations512" / "fold{:02}{}.txt".format(fold_index, mode)
        assert fp.exists()
        indices = []
        with fp.open('r') as handle:
            while True:
                line = handle.readline()
                if len(line) == 0:
                    break
                indices.append(line[:-1])
        return indices        
    
    def get_train_test_indices(self, fold_index):
        assert 1 <= fold_index <= self.n_folds
        train_indices = self._get_fold_indices(fold_index, mode='')
        test_indices = self._get_fold_indices(fold_index, mode='test')
        assert set(test_indices) & set(train_indices) == set()
        return train_indices, test_indices


def parse_target(target_str):
    splt = target_str.split(' ')
    assert len(splt) == 14, "{}".format(target_str)
    output = {
        "coords": [float(v) for v in splt[0:3]],
        "class_id": int(splt[3]),
        "is_fully_visible": bool(splt[4]),
        "is_occluded": bool(splt[5]),
        "oriented_bbox": [(int(splt[i]), int(splt[i + 4])) for i in range(6, 10)]
    }
    return output


def get_parsed_target(target_filename):
    output = []
    with Path(target_filename).open('r') as handle:
        while True:
            line = handle.readline()
            if len(line) == 0:
                break
            output.append(parse_target(line[:-1]))
    return output


class VedaiFiles512x512(Dataset):

    def __init__(self, path, mode='train', fold_index=1, img_type='co', max_n_samples=None):
        assert mode in ('train', 'test')
        self.mode = mode
        ds = _VedaiFiles(path, img_type=img_type, option='512')
        train_indices, test_indices = ds.get_train_test_indices(fold_index)
        indices = train_indices if mode == 'train' else test_indices
        if max_n_samples is not None:
            indices = indices[:max_n_samples]
        self.samples = [ds[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class VedaiFiles1024x1024(Dataset):

    def __init__(self, path, mode='train', fold_index=1, img_type='co'):
        assert mode in ('train', 'test')
        self.mode = mode
        ds = _VedaiFiles(path, img_type=img_type, option='1024')
        train_indices, test_indices = ds.get_train_test_indices(fold_index)
        indices = train_indices if mode == 'train' else test_indices
        self.samples = [ds[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


# Map VEDAI indices to aranged indices
idx2idx = {
    1: 0,
    2: 1,
    4: 2,
    5: 3,
    23: 4,
    7: 5,
    8: 6,
    9: 7,
    10: 8,
    11: 9,
    31: 10
}


class TargetToCountedLabels:

    def __init__(self, n_labels):
        self.n_labels = n_labels

    def __call__(self, image_filepath, target_filepath):
        targets = get_parsed_target(target_filepath)
        out_labels = np.zeros((self.n_labels, ), dtype=np.int)
        for t in targets:
            out_labels[idx2idx[t['class_id']]] = 1
        return image_filepath, out_labels


class DatapointToImageBoxesLabels:

    def __init__(self, n_labels):
        self.n_labels = n_labels

    def __call__(self, image_filepath, target_filepath):
        img_pil = imread_pillow(image_filepath)
        targets = get_parsed_target(target_filepath)

        boxes = []
        labels = []
        if len(targets) == 0:
            # If tile is pur negative (no detections), we recreate a random box with a negative label
            # such that it is ignored in the loss function
            w, h = img_pil.size
            xmin = random.randint(0, w - 1)
            ymin = random.randint(0, h - 1)
            xmax = random.randint(xmin, w)
            ymax = random.randint(ymin, h)
            boxes.append((xmin, ymin, xmax, ymax))
            labels.append(-1)
        else:
            for t in targets:
                xmin_y_min = np.min(t['oriented_bbox'], axis=0).astype(int)
                xmax_y_max = np.max(t['oriented_bbox'], axis=0).astype(int)
                boxes.append((int(xmin_y_min[0]), int(xmin_y_min[1]), int(xmax_y_max[0]), int(xmax_y_max[1])))
                labels.append(idx2idx[t['class_id']])
        return img_pil, (boxes, labels)


class ProxyDataset(object):

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]


class TransformedDataset(ProxyDataset):

    def __init__(self, ds, xy_transforms, return_input=False):
        super(TransformedDataset, self).__init__(ds)
        assert isinstance(xy_transforms, (tuple, list)), "xy_transforms should be a list/tuple of callable"
        for t in xy_transforms:
            assert callable(t), "Given {}".format(t)

        self.xy_transforms = xy_transforms
        self.return_input = return_input

    def __getitem__(self, idx):
        tdp = self.ds[idx]
        for t in self.xy_transforms:
            tdp = t(*tdp)
        if self.return_input:
            return tuple(tdp) + (self.ds[idx], )
        return tdp

from pathlib import Path
import sys
# Load common module
sys.path.insert(0, Path(__file__).absolute().parent.parent.as_posix())

import numpy as np

from common.dataset import TrainvalFilesDataset, read_image, TransformedDataset

train_dataset_path = "/home/fast_storage/imaterialist-challenge-furniture-2018/train"
val_dataset_path = "/home/fast_storage/imaterialist-challenge-furniture-2018/validation"

train_dataset = TrainvalFilesDataset(train_dataset_path)
val_dataset = TrainvalFilesDataset(val_dataset_path)

from tqdm import tqdm
from joblib import Parallel, delayed


def task(dp):
    try:
        img = read_image(dp[0])
        res = img.crop(box=(10, 10, 100, 100))
        del res
        del img
        return False
    except Exception as e:
        return True


with Parallel(n_jobs=15) as parallel:
    result = parallel(delayed(task)(dp) for dp in tqdm(train_dataset))

print(np.array(train_dataset.images)[result])
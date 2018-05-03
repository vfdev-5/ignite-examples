from pathlib import Path

from PIL import Image

import pandas as pd

import catboost as cat

from sklearn.model_selection import StratifiedKFold

from common.dataset import FilesFromCsvDataset, TransformedDataset
from common.meta import get_metafeatures, get_imsize_and_targets
from hyperopt import hp


DEBUG = True
SEED = 2018

OUTPUT_PATH = "output"


meta_features_path = Path("output")
meta_features_list = [
    meta_features_path / "val_probas_inceptionresnetv2_350_resized_crop" / "20180428_1622" / "probas.csv",
    meta_features_path / "val_probas_inceptionv4_350_resized_crop" / "20180428_1633" / "probas.csv",
]
meta_features = get_metafeatures(meta_features_list)

dataset = FilesFromCsvDataset("output/filtered_val_dataset.csv")
dataset = TransformedDataset(dataset,
                             transforms=lambda x: (x, Image.open(x).size),
                             target_transforms=lambda l: l - 1)
df_imsize_targets = get_imsize_and_targets(dataset)

X = pd.concat([meta_features, df_imsize_targets[['width', 'height']]], axis=1)
X.dropna(inplace=True)
X = X.values
Y = df_imsize_targets['target'].values

# Cross-validation parameters
CV_SPLIT = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

MODEL = cat.CatBoostClassifier

SCORINGS = ["neg_log_loss", ]

FIT_PARAMS = {

}

# Model parameters
MODEL_PARAMS = {
    "iterations": 5,
    "loss_function": "MultiClassOneVsAll",
    "od_type": "Iter",
    "od_wait": 50,
    "bootstrap_type": "Bernoulli",
    "task_type": "CPU",
    "verbose": True,
    "metric_period": 1
}

# Optional, hyperparameters tunning
MODEL_HP_PARAMS = {
    "depth": 2 + hp.randint("depth", 5),
    "learning_rate": hp.quniform("learning_rate", 0.001, 0.5, 0.005),
    "l2_leaf_reg": 2 + hp.randint("l2_leaf_reg", 2),
    "random_seed": hp.randint("random_seed", 12345),
    "subsample": hp.quniform("subsample", 0.5, 1.0, 0.01)
}

# Optional, hyperparameters tunning
N_TRIALS = 10

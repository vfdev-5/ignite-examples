from pathlib import Path
import json

import numpy as np
from sklearn.externals import joblib

import pandas as pd


def save_params(params, filepath):
    json.dump(params, filepath)


def load_params(filepath):
    return json.load(filepath)


def save_model(estimator, filepath):
    joblib.dump(estimator, filepath)


def load_model(filepath):
    joblib.load(filepath)


def get_metafeatures(prediction_files):
    dfs = [pd.read_csv(f, index_col='id') for f in prediction_files]
    for i, df in enumerate(dfs):
        df.columns = ["f{}_{}".format(i, c) for c in df.columns]
    meta_features = pd.concat([df for df in dfs], axis=1)
    return meta_features


def get_imsize_and_targets(dataset):
    indices = np.zeros((len(dataset)), dtype=np.int)
    targets = np.zeros((len(dataset)), dtype=np.int)
    imsizes = np.zeros((len(dataset), 2), dtype=np.int)

    def _get_index(filepath):
        stem = filepath.stem
        split = stem.split("_")
        return int(split[0])

    for i, dp in enumerate(dataset):
        indices[i] = _get_index(Path(dp[0][0]))
        imsizes[i, :] = dp[0][1]
        targets[i] = dp[1]

    df = pd.DataFrame({"width": imsizes[:, 0], "height": imsizes[:, 1], "target": targets}, index=indices)
    return df

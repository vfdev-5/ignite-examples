import json
from sklearn.externals import joblib


def save_params(params, filepath):
    json.dump(params, filepath)


def load_params(filepath):
    return json.load(filepath)


def save_model(estimator, filepath):
    joblib.dump(estimator, filepath)


def load_model(filepath):
    joblib.load(filepath)

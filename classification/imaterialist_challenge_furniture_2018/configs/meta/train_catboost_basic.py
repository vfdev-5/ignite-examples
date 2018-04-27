from pathlib import Path
import catboost as cat

from sklearn.model_selection import StratifiedKFold

from common.dataset import FilesFromCsvDataset

DEBUG = True
SEED = 2018

OUTPUT_PATH = "output"
MF_DATASET_PATH = Path("output")


META_FEATURES_LIST = [
    MF_DATASET_PATH / "val_inference_FurnitureInceptionResNet350_12345" / "probas.csv",
    MF_DATASET_PATH / "val_inference_FurnitureInceptionV4350_12345" / "probas.csv",
    MF_DATASET_PATH / "val_inference_FurnitureNASNet350_12345" / "probas.csv",
    MF_DATASET_PATH / "val_inference_FurnitureDenseNet350_12345" / "probas.csv",
]

TRAIN_DATASET = FilesFromCsvDataset("output/filtered_val_dataset.csv")

# Cross-validation parameters
CV_SPLIT = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

MODEL = cat.CatBoostClassifier

SCORINGS = ["neg_log_loss", "precision_macro", "recall_macro"]

FIT_PARAMS = {

}

# Model parameters
MODEL_PARAMS = {

}

# Optional, hyperparameters tunning
# MODEL_HP_PARAMS = {
#
# }

# Optional, hyperparameters tunning
N_TRIALS = 10

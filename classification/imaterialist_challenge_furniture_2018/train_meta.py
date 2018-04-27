from pathlib import Path
import sys
from argparse import ArgumentParser
import random
import logging
from importlib import util

import numpy as np

from sklearn.model_selection import cross_validate

from hyperopt import fmin, tpe, hp, tpe, STATUS_OK, Trials


# Load common module
sys.path.insert(0, Path(__file__).absolute().parent.parent.as_posix())
from common import setup_logger, save_conf
from common.meta import save_params, save_model


def load_config(config_filepath):
    assert Path(config_filepath).exists(), "Configuration file '{}' is not found".format(config_filepath)
    # Load custom module
    spec = util.spec_from_file_location("config", config_filepath)
    custom_module = util.module_from_spec(spec)
    spec.loader.exec_module(custom_module)
    config = custom_module.__dict__
    assert "META_FEATURES_LIST" in config, "META_FEATURES_LIST parameter is not found in configuration file"

    assert "TRAIN_DATASET" in config, "TRAIN_DATASET parameter is not found in configuration file"

    assert "OUTPUT_PATH" in config, "OUTPUT_PATH is not found in the configuration file"

    assert "SCORINGS" in config, "SCORINGS parameter should be specified in the configuration file"
    assert isinstance(config["SCORINGS"], (list, tuple)), "SCORINGS should be a list or tuple of strings"

    if "FIT_PARAMS" not in config:
        config["FIT_PARAMS"] = {}

    assert "MODEL" in config, "MODEL is not found in configuration file"
    assert isinstance(config["MODEL"], type), "MODEL should be a class to instantiate an object"
    assert "MODEL_PARAMS" in config, "MODEL_PARAMS is not found in configuration file"
    assert isinstance(config["MODEL_PARAMS"], dict)

    if "MODEL_HP_PARAMS" not in config:
        config["MODEL_HP_PARAMS"] = {}

    if "N_TRIALS" not in config:
        config["N_TRIALS"] = 1

    if "N_JOBS" not in config:
        config["N_JOBS"] = 1

    return config


import pandas as pd


def get_metafeatures(prediction_files):
    dfs = [pd.read_csv(f, index_col='id') for f in prediction_files]
    names = ["f{}".format(i) for i in range(len(prediction_files))]
    meta_features = pd.concat([df for df in dfs], axis=1, names=names)
    return meta_features


def hp_optimize(score_fn, params_space, max_evals):
    trials = Trials()
    best_params = fmin(score_fn, params_space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
    return best_params, trials


def run(config_file):
    print("--- iMaterialist 2018 : Meta-learner training --- ")

    print("Load config file ... ")
    config = load_config(config_file)

    seed = config.get("SEED", 2018)
    random.seed(seed)

    output = Path(config["OUTPUT_PATH"])
    model = config["MODEL"]
    model_name = model.__class__.__name__
    debug = config.get("DEBUG", False)

    from datetime import datetime
    now = datetime.now()
    log_dir = output / ("training_meta_{}_{}".format(model_name, now.strftime("%Y%m%d_%H%M")))
    assert not log_dir.exists(), \
        "Output logging directory '{}' already existing".format(log_dir)
    log_dir.mkdir(parents=True)

    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
        print("Activated debug mode")

    logger = logging.getLogger("iMaterialist 2018: Train meta-learner")
    setup_logger(logger, (log_dir / "train.log").as_posix(), log_level)

    save_conf(config_file, log_dir.as_posix(), logger)

    X = None
    y = None

    n_trials = config["N_TRIALS"]
    scorings = config["SCORINGS"]
    cv = config["CV_SPLIT"]
    estimator_cls = config["MODEL"]
    model_params = config["MODEL_PARAMS"]
    model_hp_params = config["MODEL_HP_PARAMS"]
    model_hp_params.update(model_params)
    fit_params = config["FIT_PARAMS"]
    n_jobs = config["N_JOBS"]

    def hp_score(model_hp_params):

        estimator = estimator_cls(**model_hp_params)

        scores = cross_validate(estimator, X, y, cv=cv, scoring=scorings,
                                fit_params=fit_params,
                                n_jobs=n_jobs, verbose=debug)

        logger.info("CV scores:")
        for scoring in scorings:
            logger.info("{} : {}".format(scoring, scores[scoring].tolist()))

        return {
            'loss': scores[scorings[0]],
            'status': STATUS_OK
        }

    logger.debug("Start training: {} epochs".format(n_trials))
    try:
        best_params, trials = hp_optimize(hp_score, model_hp_params, max_evals=n_trials)
        best_params.update(model_params)

        save_params(best_params, (log_dir / "best_params.json").as_posix())

        logger.info("Best parameters: \n{}".format(best_params))
        logger.info("Best trial : \n{}".format(trials.best_trial))

        logger.info("Train meta model on complete dataset")
        estimator = estimator_cls(**best_params)

        save_model(estimator, (log_dir / "best_model.pkl").as_posix())

    except KeyboardInterrupt:
        logger.info("Catched KeyboardInterrupt -> exit")
    except Exception as e:  # noqa
        logger.exception("")
        if debug:
            try:
                # open an ipython shell if possible
                import IPython
                IPython.embed()  # noqa
            except ImportError:
                print("Failed to start IPython console")

    logger.debug("Training is ended")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("config_file", type=str, help="Configuration file. See examples in configs/meta")
    args = parser.parse_args()
    run(args.config_file)

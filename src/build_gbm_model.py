import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
import numpy as np
from numpy import ndarray, arange, argmin
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.fmin import generate_trials_to_calculate

from src.compare_predictions import RegressionMetrics, evaluate_regression
from src.split_data import TRAIN_DATA_PATH, TUNE_DATA_PATH
from src.xgb_prepare_data import prepare_data

TARGET_COLUMN = "Y"
FEATURE_COLUMNS: list[str] = [
    "AGE",
    "SEX_NUM",
    "BMI",
    "BP",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
]
WEIGHT_COLUMN = None
RANDOM_SEED: int = 422134987614
MODEL_PARAMS: dict = {
    "objective": "reg:squarederror",
    "device": "gpu",
    "tree_method": "hist",
    "learning_rate": 0.1,
    "max_depth": hp.choice("max_depth", arange(1, 7, 1)),
    "max_bin": hp.choice("max_bin", arange(2, 256, 1)),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.8, 1),
    "colsample_bylevel": hp.uniform("colsample_bylevel", 0.8, 1),
    "subsample": hp.uniform("subsample", 0.5, 1),
    "gamma": hp.uniform("gamma", 0, 0.2),
    "min_child_weight": hp.choice("min_child_weight", arange(1, 10, 1)),
    "seed": RANDOM_SEED,
    "num_boost_round": 100,
    "early_stopping_rounds": 10,
}

PARAM_TRIALS: int = 100

XGB_MODEL_FILENAME = "data/xgboost_model/model.ubj"


def getBestModelfromTrials(trials):
    valid_trial_list = [
        trial for trial in trials if STATUS_OK == trial["result"]["status"]
    ]
    losses = [float(trial["result"]["loss"]) for trial in valid_trial_list]
    index_having_minumum_loss: int = argmin(losses)
    return valid_trial_list[index_having_minumum_loss]["result"]["model"]


def score_model(model: xgb.Booster, data: xgb.DMatrix) -> RegressionMetrics:

    predictions: ndarray = model.predict(data)
    y: ndarray = data.get_label()
    weights: ndarray = data.get_weight()
    if weights.size == 0:
        weights = None

    return evaluate_regression(targets=y, predictions=predictions, weights=weights)


def build_xgboost_model(
    train_data: xgb.DMatrix,
    tune_data: xgb.DMatrix = None,
    params: dict = MODEL_PARAMS,
    param_trials: int = PARAM_TRIALS,
):

    trials = Trials()
    # Include best params so far

    trials = generate_trials_to_calculate(
        [
            {
                "objective": "reg:squarederror",
                "device": "gpu",
                "tree_method": "hist",
                "learning_rate": 0.1,
                "seed": RANDOM_SEED,
                "num_boost_round": 100,
                "early_stopping_rounds": 10,
            }
            | {
                "colsample_bylevel": np.float64(0.9758194276451145),
                "colsample_bytree": np.float64(0.9672745551883308),
                "gamma": np.float64(0.07924871129549596),
                "max_bin": np.int64(6),
                "max_depth": np.int64(2),
                "min_child_weight": np.int64(5),
                "subsample": np.float64(0.9767236964288231),
            }
        ]
    )

    def objective(params: dict) -> dict:
        model: xgb.Booster = xgb.train(
            params,
            train_data,
            evals=[(train_data, "train"), (tune_data, "tune")],
            num_boost_round=params["num_boost_round"],
            early_stopping_rounds=params["early_stopping_rounds"],
        )

        tune_results = score_model(model, tune_data)

        return {"loss": tune_results.rmse, "status": STATUS_OK, "model": model}

    best_params = fmin(
        fn=objective,
        space=params,
        algo=tpe.suggest,
        max_evals=param_trials,
        trials=trials,
    )

    print(f"Best Parameters: {best_params}")

    model = getBestModelfromTrials(trials)

    # Score the model
    train_results = score_model(model, train_data)
    print(f"Training Results: {train_results}")
    tune_results = score_model(model, tune_data)
    print(f"Tuning Results: {tune_results}")

    # Save the model
    model_path: Path = Path(XGB_MODEL_FILENAME).parent
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_model(XGB_MODEL_FILENAME)
    print(f"Model saved to {model_path}")

    # Save the metrics
    with open(model_path / "metrics.txt", "w") as f:
        f.write(tune_results.__repr__())

    return trials


if __name__ == "__main__":

    train_tune: list[xgb.DMatrix] = []
    for path in [TRAIN_DATA_PATH, TUNE_DATA_PATH]:
        train_tune.append(
            prepare_data(
                df=pd.read_parquet(path),
                feature_columns=FEATURE_COLUMNS,
                target_column=TARGET_COLUMN,
                weight_column=WEIGHT_COLUMN,
            )
        )

    trials = build_xgboost_model(train_data=train_tune[0], tune_data=train_tune[1])

from pathlib import Path
import json

from numpy import ndarray

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

from src.compare_predictions import RegressionMetrics, evaluate_regression
from src.features import add_features
from src.xgb_prepare_data import prepare_data

TARGET_COLUMN = "Y"
FEATURE_COLUMNS = ["AGE", "SEX_NUM", "BMI", "BP", "S1", "S2", "S3", "S4", "S5", "S6"]
WEIGHT_COLUMN = None
RANDOM_SEED: int = 422134987614
MODEL_PARAMS: dict = {
    "objective": "reg:squarederror",
    "device": "gpu",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.1,
    "max_bin": 32,
    "colsample_bytree": 1,
    "colsample_bylevel": 1,
    "subsample": 0.8,
    "seed": RANDOM_SEED,
    "num_boost_round": 100,
    "early_stopping_rounds": 10,
}

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
):

    model: xgb.Booster = xgb.train(
        params,
        train_data,
        evals=[(train_data, "train"), (tune_data, "tune")],
        num_boost_round=params["num_boost_round"],
        early_stopping_rounds=params["early_stopping_rounds"],
    )

    # Score the model
    train_results = score_model(model, train_data)
    print(f"Training Results: {train_results}")
    tune_results = score_model(model, tune_data)
    print(f"Tuning Results: {tune_results}")

    # Save the model
    model_path = Path("data/xgboost_model/")
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path / "model.ubj")
    print(f"Model saved to {model_path}")

    # Save the metrics
    with open(model_path / "metrics.txt", "w") as f:
        f.write(tune_results.__repr__())


if __name__ == "__main__":

    train_data_path = "data/train.parquet"
    tune_data_path = "data/tune.parquet"

    train_tune: list[xgb.DMatrix] = []
    for path in [train_data_path, tune_data_path]:
        train_tune.append(prepare_data(df=add_features(pd.read_parquet(path)), feature_columns=FEATURE_COLUMNS, target_column=TARGET_COLUMN, weight_column=WEIGHT_COLUMN))

    build_xgboost_model(
        train_data=train_tune[0], tune_data=train_tune[1], params=MODEL_PARAMS
    )

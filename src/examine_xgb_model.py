from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from matplotlib import pyplot as plt

from src.build_gbm_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    WEIGHT_COLUMN,
    XGB_MODEL_FILENAME,
)
from src.split_data import TUNE_DATA_PATH
from src.xgb_prepare_data import prepare_data


def examine_xgb_model(
    model: xgb.Booster,
    tune_data: pd.DataFrame,
    feature_columns=FEATURE_COLUMNS,
    target_column=TARGET_COLUMN,
    weight_column=WEIGHT_COLUMN,
) -> pd.DataFrame:
    """Examine the model.

    Args:
        model (xgb.Booster): The model to examine.
        tune_data (xgb.DMatrix): The data to use for examination.

    Returns:
        pd.DataFrame: The examination results.
    """
    tune_matrix = prepare_data(
        df=tune_data,
        feature_columns=feature_columns,
        target_column=target_column,
        weight_column=weight_column,
    )

    tune_data["prediction"] = model.predict(tune_matrix)
    tune_data["residual"] = tune_data[target_column] - tune_data["prediction"]
    tune_data["residual_abs"] = tune_data["residual"].abs()

    # Plotting path
    plot_folder = Path("results/xgboost_model")
    plot_folder.mkdir(parents=True, exist_ok=True)

    # Actual vs. Predicted
    tune_data[[target_column, "prediction"]].plot.scatter(
        x=target_column, y="prediction"
    )
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual - Tune Data")

    with open(plot_folder / "pred_vs_actual.png", "wb") as file:
        plt.savefig(file)

    plt.clf()

    # Seaborn pairplot
    # sns.pairplot(tune_data, hue="residual_abs", diag_kind="kde")
    # plt.title("Tune Data Pairplot")

    # filepath = Path("results/xgboost_model/pairplot.png")
    # with open(filepath, "wb") as file:
    #     plt.savefig(file)

    # Feature Importance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(tune_matrix.get_data())

    shap.summary_plot(shap_values, tune_data[feature_columns], plot_type="bar")
    plt.title("Feature Importance - Tune Data")
    plt.ylabel("Feature")

    filepath = Path("results/xgboost_model/feature_importance_bar.png")
    with open(plot_folder / "feature_importance_bar.png", "wb") as file:
        plt.savefig(file)
    plt.clf()

    shap.summary_plot(shap_values, tune_data[feature_columns])
    plt.title("Feature Importance - Tune Data")
    plt.ylabel("Feature")

    filepath = Path("results/xgboost_model/feature_importance.png")
    with open(filepath, "wb") as file:
        plt.savefig(file)
    plt.clf()

    return


if __name__ == "__main__":
    model = xgb.Booster()
    model.load_model(XGB_MODEL_FILENAME)

    tune_data = pd.read_parquet(TUNE_DATA_PATH)
    examine_xgb_model(model, tune_data)

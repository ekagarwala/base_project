import pandas as pd
import xgboost as xgb

from src.features import add_features

def prepare_data(
    df: pd.DataFrame,
    feature_columns: list,
    target_column: str = None,
    weight_column: str = None,
) -> xgb.DMatrix:
    df = add_features(df)

    X: pd.DataFrame = df[feature_columns]
    y: pd.Series = df[target_column] if target_column else None
    weights: pd.Series = df[weight_column] if weight_column else None

    return xgb.DMatrix(
        data=X,
        label=y,
        feature_names=feature_columns,
        weight=weights,
    )
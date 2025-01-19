import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["SEX_NUM"] = df["SEX"].map({"M": 0, "F": 1}).astype(int)
    # print(df.dtypes)
    # print(df.head())
    return df

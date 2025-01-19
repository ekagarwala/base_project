import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.6
TUNE_SIZE = 0.2
TEST_SIZE = 0.2

KEY_ID = ""
STRATIFY_COLS = []

RANDOM_STATE = 234551


def split_dataframe(
    df,
    train_size=TRAIN_SIZE,
    tune_size=TUNE_SIZE,
    test_size=TEST_SIZE,
    key_id: str = KEY_ID,
    stratify_cols: list[str] = STRATIFY_COLS,
    random_state=RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if train_size + tune_size + test_size != 1.0:
        raise ValueError("train_size, tune_size, and test_size must sum to 1.0.")
    if min(train_size, tune_size, test_size) <= 0:
        raise ValueError(
            "train_size, tune_size, and test_sizes must be greater than 0."
        )
    if min(train_size, tune_size, test_size) * df.shape[0] < 1:
        raise ValueError(
            f"train_size, tune_size, and test_size must be greater than {1/df.shape[0]} given e {df.shape[0]} rows."
        )

    # Create stratify variable
    if stratify_cols:
        stratify = df[stratify_cols]
    else:
        stratify = None

    # Check if key_ids are present across multiple stratify groups
    if key_id:
        df_key_stratify = df[[key_id] + stratify_cols].drop_duplicates()

        # if stratify_cols are not provided, this check always passes
        if df_key_stratify.shape[0] != df_key_stratify.key_id.nunique():
            raise ValueError(
                "Some key_ids are represented in multiple stratify groups."
            )

        train_keys, temp_keys = train_test_split(
            df_key_stratify,
            train_size=train_size,
            stratify=stratify,
            random_state=random_state,
        )
        tune_keys, test_keys = train_test_split(
            temp_keys,
            train_size=tune_size / (tune_size + test_size),
            stratify=stratify,
            random_state=random_state,
        )

        train_df = df[df[key_id].isin(train_keys)]
        tune_df = df[df[key_id].isin(tune_keys)]
        test_df = df[df[key_id].isin(test_keys)]

        return train_df, tune_df, test_df

    train_df, temp_df = train_test_split(
        df, train_size=train_size, stratify=stratify, random_state=random_state
    )
    tune_df, test_df = train_test_split(
        temp_df,
        train_size=tune_size / (tune_size + test_size),
        stratify=stratify,
        random_state=random_state,
    )

    return train_df, tune_df, test_df


if __name__ == "__main__":
    df = pd.read_parquet("data/base_data.parquet")
    train_df, tune_df, test_df = split_dataframe(df)
    train_df.to_parquet("data/train.parquet")
    tune_df.to_parquet("data/tune.parquet")
    test_df.to_parquet("data/test.parquet")

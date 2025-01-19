import argparse

import pandas as pd
import pyarrow

gender_cat: pd.CategoricalDtype = pd.CategoricalDtype(["M", "F"], ordered=True)


def import_csv_to_dataframe(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, dtype={"SEX": gender_cat})
    print(df.dtypes)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import CSV to DataFrame and save as Parquet"
    )
    parser.add_argument("csv_file_path", type=str, help="Path to the CSV file")
    args = parser.parse_args()

    csv_file_path = args.csv_file_path
    parquet_file_path = "data/base_data.parquet"

    df = import_csv_to_dataframe(csv_file_path)
    df.to_parquet(parquet_file_path)

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_sms(raw_path) -> None:
    """Clean and rename the dataset and save it in data/processed"""
    Path("/opt/static/raw").mkdir(parents=True, exist_ok=True)
    Path("/opt/static/processed").mkdir(parents=True, exist_ok=True)

    # Load dataset
    df = pd.read_csv(raw_path)

    # Clean dataset
    df = df.drop_duplicates(keep="first")

    # Rename
    df["label"] = df.label.map({"ham": 0, "spam": 1})

    # Preprocessing
    df = df.dropna()

    # Save
    df.to_csv("/opt/static/processed/dataset.csv", index=False)


def train_val_test_split(
    df: pd.DataFrame, train_size: float = 0.8, has_val: bool = True
):
    """Return a tuple (DataFrame, DatasetDict) with a custom train/val/split"""

    # train/val/test split
    df_train, df_test = train_test_split(
        df, test_size=1 - train_size, shuffle=True, stratify=df["label"]
    )

    if has_val:
        df_test, df_val = train_test_split(
            df_test, test_size=0.5, stratify=df_test["label"]
        )
        return df_train, df_val, df_test

    else:
        return df_train, df_test

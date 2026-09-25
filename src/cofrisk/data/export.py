from pathlib import Path

import pandas as pd
from scipy.io import arff

from cofrisk.data.cleaning import (
    encode_target,
    remove_exact_duplicates,
)
from cofrisk.data.validation import validate_dataset


RAW_FILE = Path("data/raw/1year.arff")
PROCESSED_DIR = Path("data/processed")
PROCESSED_FILE = PROCESSED_DIR / "bankruptcy.parquet"


def load_arff(file_path: Path) -> pd.DataFrame:
    data, _ = arff.loadarff(file_path)
    return pd.DataFrame(data)


def prepare_dataset() -> pd.DataFrame:
    df = load_arff(RAW_FILE)

    validate_dataset(df)

    df = remove_exact_duplicates(df)
    df = encode_target(df)

    return df


def export_parquet() -> None:
    df = prepare_dataset()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        PROCESSED_FILE,
        index=False,
    )

    print(f"Parquet exported to: {PROCESSED_FILE}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    export_parquet()
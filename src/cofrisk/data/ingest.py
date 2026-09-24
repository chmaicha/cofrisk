from pathlib import Path

import boto3
import pandas as pd
from scipy.io import arff


BUCKET_NAME = "cofrisk-chaimae"
S3_KEY = "raw/bankruptcy/1year.arff"

LOCAL_DIR = Path("data/raw")
LOCAL_FILE = LOCAL_DIR / "1year.arff"


def download_from_s3() -> Path:
    """Download the raw dataset from S3."""
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")

    s3.download_file(
        BUCKET_NAME,
        S3_KEY,
        str(LOCAL_FILE),
    )

    return LOCAL_FILE


def load_arff(file_path: Path) -> pd.DataFrame:
    """Load an ARFF file into a Pandas DataFrame."""
    data, metadata = arff.loadarff(file_path)

    df = pd.DataFrame(data)

    return df


def profile_dataset(df: pd.DataFrame) -> None:
    """Print basic dataset profiling information."""

    print("\n===== SHAPE =====")
    print(df.shape)

    print("\n===== COLUMNS =====")
    print(df.columns.tolist())

    print("\n===== DTYPES =====")
    print(df.dtypes)

    print("\n===== FIRST ROWS =====")
    print(df.head())

    print("\n===== MISSING VALUES =====")
    print(df.isna().sum())

    print("\n===== DUPLICATES =====")
    print(df.duplicated().sum())

    print("\n===== TARGET DISTRIBUTION =====")
    print(df["class"].value_counts(dropna=False))

    print("\n===== TARGET PROPORTIONS =====")
    print(df["class"].value_counts(normalize=True, dropna=False))


if __name__ == "__main__":
    file_path = download_from_s3()

    print(f"Dataset downloaded to: {file_path}")

    df = load_arff(file_path)

    profile_dataset(df)
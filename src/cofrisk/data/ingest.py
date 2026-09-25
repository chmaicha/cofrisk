from pathlib import Path
from cofrisk.data.eda import run_eda
import boto3
import pandas as pd
from scipy.io import arff

from cofrisk.data.cleaning import (
    encode_target,
    remove_exact_duplicates,
)
from cofrisk.data.validation import validate_dataset


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


if __name__ == "__main__":

    # 1. Ingestion
    file_path = download_from_s3()

    print(f"Dataset downloaded to: {file_path}")

    # 2. Loading
    df = load_arff(file_path)

    # 3. Validation
    validate_dataset(df)

    # 4. Remove exact duplicates
    df = remove_exact_duplicates(df)

    # 5. Encode target
    df = encode_target(df)
    run_eda(df)
    # print("\n===== CLEANED DATASET =====")
    # print(f"Shape: {df.shape}")

    # print("\n===== DTYPES =====")
    # print(df.dtypes.tail())

    # print("\n===== TARGET DISTRIBUTION =====")
    # print(df["class"].value_counts())

    # print("\n===== TARGET TYPE =====")
    # print(df["class"].dtype)
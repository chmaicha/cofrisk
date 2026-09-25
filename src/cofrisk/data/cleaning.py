import pandas as pd

from cofrisk.data.validation import TARGET_COLUMN


def remove_exact_duplicates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove exact duplicate observations."""

    before = len(df)

    cleaned_df = df.drop_duplicates(
        keep="first"
    ).copy()

    removed = before - len(cleaned_df)

    print("\n===== DUPLICATE REMOVAL =====")
    print(f"Rows before: {before}")
    print(f"Rows after:  {len(cleaned_df)}")
    print(f"Rows removed: {removed}")

    return cleaned_df


def encode_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert the binary target from bytes to integers."""

    cleaned_df = df.copy()

    cleaned_df[TARGET_COLUMN] = (
        cleaned_df[TARGET_COLUMN]
        .map({
            b"0": 0,
            b"1": 1,
        })
        .astype("int8")
    )

    return cleaned_df
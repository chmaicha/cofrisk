import pandas as pd
from sklearn.model_selection import train_test_split

from cofrisk.data.validation import (
    EXPECTED_FEATURES,
    TARGET_COLUMN,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20


def split_dataset(
    df: pd.DataFrame,
):
    """Split dataset into stratified train and test sets."""

    X = df[EXPECTED_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\n===== TRAIN / TEST SPLIT =====")

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")

    print("\nTrain target distribution:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True).mul(100).round(2))

    print("\nTest target distribution:")
    print(y_test.value_counts())
    print(y_test.value_counts(normalize=True).mul(100).round(2))

    return X_train, X_test, y_train, y_test
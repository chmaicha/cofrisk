import pandas as pd


EXPECTED_FEATURES = [f"Attr{i}" for i in range(1, 65)]
TARGET_COLUMN = "class"
EXPECTED_COLUMNS = EXPECTED_FEATURES + [TARGET_COLUMN]

EXPECTED_TARGET_VALUES = {b"0", b"1"}


def report_iqr_outliers(df: pd.DataFrame) -> None:
    """Report the number of IQR-based outliers per feature."""

    print("\n===== IQR OUTLIERS =====")

    rows = []

    for column in EXPECTED_FEATURES:
        series = df[column].dropna()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        rows.append(
            {
                "feature": column,
                "outlier_count": outliers.sum(),
                "outlier_percentage": (
                    outliers.mean() * 100
                ),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "outlier_percentage",
        ascending=False,
    )

    print(result.to_string(index=False))

def validate_columns(df: pd.DataFrame) -> None:
    """Validate the dataset schema."""

    actual_columns = df.columns.tolist()

    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Invalid dataset schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Received: {actual_columns}"
        )


def validate_features_are_numeric(df: pd.DataFrame) -> None:
    """Ensure all financial features are numeric."""

    non_numeric = df[EXPECTED_FEATURES].select_dtypes(
        exclude="number"
    ).columns.tolist()

    if non_numeric:
        raise TypeError(
            f"Non-numeric features detected: {non_numeric}"
        )


def validate_target(df: pd.DataFrame) -> None:
    """Validate the target column."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Target column 'class' is missing.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values.")

    actual_values = set(df[TARGET_COLUMN].unique())

    if not actual_values.issubset(EXPECTED_TARGET_VALUES):
        raise ValueError(
            "Unexpected target values.\n"
            f"Expected: {EXPECTED_TARGET_VALUES}\n"
            f"Received: {actual_values}"
        )


def report_missing_values(df: pd.DataFrame) -> None:
    """Report missing values."""

    missing = df.isna().sum()
    missing = missing[missing > 0]

    print("\n===== MISSING VALUES =====")

    if missing.empty:
        print("No missing values.")
        return

    result = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_percentage": (
                missing / len(df) * 100
            ).round(2),
        }
    )

    print(result)

def analyze_missingness_by_target(df: pd.DataFrame) -> None:
    """Compare missing values between target classes."""

    print("\n===== MISSINGNESS BY TARGET =====")

    result = (
        df.groupby(TARGET_COLUMN)[EXPECTED_FEATURES]
        .apply(lambda group: group.isna().mean() * 100)
        .T
    )

    result.columns = [
        f"class_{str(column)}"
        for column in result.columns
    ]

    print(result.sort_values(
        by=result.columns.tolist(),
        ascending=False
    ).head(15))

def report_extreme_values(df: pd.DataFrame) -> None:
    """Report basic extreme-value statistics."""

    print("\n===== EXTREME VALUES =====")

    stats = df[EXPECTED_FEATURES].describe().T

    result = stats[
        ["min", "25%", "50%", "75%", "max"]
    ]

    print(result)

def report_duplicates(df: pd.DataFrame) -> None:
    """Report duplicated rows."""

    duplicate_count = df.duplicated().sum()

    print("\n===== DUPLICATES =====")
    print(f"Duplicate rows: {duplicate_count}")

def analyze_duplicates(df: pd.DataFrame) -> None:
    """Analyze duplicated rows and check for conflicting targets."""

    duplicated = df[df.duplicated(keep=False)].copy()

    print("\n===== DUPLICATE ANALYSIS =====")
    print(f"Duplicated rows: {len(duplicated)}")

    if duplicated.empty:
        print("No duplicated rows.")
        return

    target_variants = (
        duplicated
        .groupby(EXPECTED_FEATURES, dropna=False)[TARGET_COLUMN]
        .nunique()
    )

    conflicting = target_variants[target_variants > 1]

    print(f"Duplicate groups: {len(target_variants)}")
    print(f"Conflicting duplicate groups: {len(conflicting)}")

def validate_dataset(df: pd.DataFrame) -> None:
    """Run all dataset validation checks."""

    print("\n===== DATASET VALIDATION =====")

    validate_columns(df)
    print("✓ Schema is valid")

    validate_features_are_numeric(df)
    print("✓ Features are numeric")

    validate_target(df)
    print("✓ Target is valid")

    report_missing_values(df)
    report_duplicates(df)
    analyze_missingness_by_target(df)
    analyze_duplicates(df)
    report_extreme_values(df)

    report_iqr_outliers(df)
    print("\n✓ Validation completed")


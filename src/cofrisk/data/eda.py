import pandas as pd

from cofrisk.data.validation import EXPECTED_FEATURES, TARGET_COLUMN


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing values by feature."""

    result = pd.DataFrame(
        {
            "missing_count": df[EXPECTED_FEATURES].isna().sum(),
            "missing_percentage": (
                df[EXPECTED_FEATURES].isna().mean() * 100
            ).round(2),
        }
    )

    result = result.sort_values(
        "missing_percentage",
        ascending=False,
    )

    print("\n===== MISSING VALUES ANALYSIS =====")
    print(result.to_string())

    return result

def analyze_target_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze target class distribution."""

    counts = df[TARGET_COLUMN].value_counts()
    percentages = (
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    result = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages,
        }
    )

    print("\n===== TARGET DISTRIBUTION =====")
    print(result)

    return result

def analyze_missingness_by_target(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze missingness rate by target class."""

    result = []

    for column in EXPECTED_FEATURES:

        missing_rate = (
            df.groupby(TARGET_COLUMN)[column]
            .apply(lambda x: x.isna().mean() * 100)
        )

        result.append(
            {
                "feature": column,
                "missing_class_0": missing_rate.get(0, 0),
                "missing_class_1": missing_rate.get(1, 0),
            }
        )

    result = pd.DataFrame(result)

    result["absolute_difference"] = (
        result["missing_class_1"]
        - result["missing_class_0"]
    ).abs()

    result = result.sort_values(
        "absolute_difference",
        ascending=False,
    )

    print("\n===== MISSINGNESS BY TARGET =====")
    print(result.head(15).to_string(index=False))

    return result
def analyze_skewness(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze feature skewness."""

    result = pd.DataFrame(
        {
            "skewness": df[EXPECTED_FEATURES].skew(),
        }
    )

    result = result.sort_values(
        "skewness",
        key=lambda x: x.abs(),
        ascending=False,
    )

    print("\n===== FEATURE SKEWNESS =====")
    print(result.head(20).to_string())

    return result

def run_eda(df: pd.DataFrame) -> None:
    """Run basic exploratory data analysis."""

    print("\n==============================")
    print("        EDA")
    print("==============================")

    analyze_target_distribution(df)
    analyze_missing_values(df)
    analyze_missingness_by_target(df)
    analyze_skewness(df)
    
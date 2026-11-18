"""Reusable feature engineering for the House Price Prediction project."""

import pandas as pd

TARGET_COLUMN = "price"
BINARY_COLUMNS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]
CATEGORICAL_COLUMNS = ["furnishingstatus"]
NUMERIC_COLUMNS = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
]
REQUIRED_FEATURES = NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready raw features plus deterministic engineered features.

    The function does not learn statistics from the data, so it is safe to call
    separately for training data and for new customer-provided house details.
    Encoding and scaling are learned later inside the sklearn Pipeline.
    """
    missing = [column for column in REQUIRED_FEATURES if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    features = data[REQUIRED_FEATURES].copy()

    for column in BINARY_COLUMNS:
        normalized = features[column].astype(str).str.strip().str.lower()
        mapped = normalized.map({"yes": 1, "no": 0, "1": 1, "0": 0})
        if mapped.isna().any():
            bad_values = sorted(normalized[mapped.isna()].unique().tolist())
            raise ValueError(f"Unexpected values in '{column}': {bad_values}")
        features[column] = mapped.astype(int)

    features["bed_bath_count"] = features["bedrooms"] + features["bathrooms"]
    denominator = features["bed_bath_count"].replace(0, 1)
    features["area_per_bed_bath"] = features["area"] / denominator
    return features

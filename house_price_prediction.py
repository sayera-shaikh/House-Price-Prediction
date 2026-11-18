"""House Price Prediction using Linear Regression and Random Forest.

Run from this folder with:
    python house_price_prediction.py

The script creates evaluation results, visualizations, and a saved sklearn
Pipeline that can predict prices from the original house-feature columns.
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    prepare_features,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Housing.csv"
PLOTS_DIR = BASE_DIR / "plots"
MODEL_PATH = BASE_DIR / "house_price_model.joblib"
RESULTS_PATH = BASE_DIR / "model_results.csv"
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS + BINARY_COLUMNS + [
                "bed_bath_count", "area_per_bed_bath"
            ]),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def make_models() -> dict:
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(scale_numeric=True)),
                ("regressor", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(scale_numeric=False)),
                ("regressor", RandomForestRegressor(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    min_samples_leaf=2,
                    n_jobs=-1,
                )),
            ]
        ),
    }


def calculate_metrics(y_true, predictions) -> dict:
    return {
        "R2": r2_score(y_true, predictions),
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
    }


def save_eda_plots(data: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.histplot(data=data, x="price", kde=True, color="steelblue")
    plt.title("Distribution of House Prices")
    plt.xlabel("Price (dataset units; currency not verified)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "price_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x="area", y="price", hue="furnishingstatus", alpha=0.7, palette="deep")
    plt.title("Area vs. House Price by Furnishing Status")
    plt.xlabel("Area")
    plt.ylabel("Price (dataset units)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "area_vs_price.png", dpi=160)
    plt.close()

    numeric_data = data.select_dtypes(include=[np.number])
    plt.figure(figsize=(9, 7))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Correlation Heatmap (Numeric Columns)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()


def save_model_plots(best_name: str, y_true, predictions, rf_pipeline) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, predictions, alpha=0.65)
    lower = min(float(np.min(y_true)), float(np.min(predictions)))
    upper = max(float(np.max(y_true)), float(np.max(predictions)))
    plt.plot([lower, upper], [lower, upper], "r--", linewidth=2, label="Perfect prediction")
    plt.xlabel("Actual Price (dataset units)")
    plt.ylabel("Predicted Price (dataset units)")
    plt.title(f"{best_name}: Actual vs. Predicted (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()

    # Random Forest feature importance is shown as model interpretation, not
    # as proof that Random Forest is the most accurate model.
    rf_preprocessor = rf_pipeline.named_steps["preprocessor"]
    transformed_names = rf_preprocessor.get_feature_names_out()
    importances = pd.Series(
        rf_pipeline.named_steps["regressor"].feature_importances_,
        index=transformed_names,
    ).sort_values(ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    sns.barplot(x=importances.values, y=importances.index, color="steelblue")
    plt.title("Top Feature Importances (Random Forest)")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=160)
    plt.close()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH.name}. Keep Housing.csv in the project folder."
        )

    data = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {data.shape}")
    print("\nFirst five rows:")
    print(data.head())
    print("\nMissing values by column:")
    print(data.isna().sum())
    print(f"\nDuplicate rows: {data.duplicated().sum()}")

    data = data.drop_duplicates().copy()
    if data["price"].isna().any():
        raise ValueError("The target column 'price' contains missing values.")

    save_eda_plots(data)

    X_raw = prepare_features(data)
    y = data["price"].astype(float)

    # 60% train, 20% validation, 20% final test.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=RANDOM_STATE
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE
    )

    models = make_models()
    fitted_models = {}
    validation_rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        validation_predictions = model.predict(X_valid)
        metrics = calculate_metrics(y_valid, validation_predictions)
        validation_rows.append({"Model": name, **{f"Validation_{k}": v for k, v in metrics.items()}})
        print(f"\n{name} validation metrics: {metrics}")

    validation_results = pd.DataFrame(validation_rows).sort_values("Validation_RMSE")
    best_name = validation_results.iloc[0]["Model"]
    print(f"\nSelected model based on validation RMSE: {best_name}")

    # Refit both models on train + validation data, then evaluate once on test.
    final_rows = []
    final_models = {}
    for name, model in models.items():
        model.fit(X_train_val, y_train_val)
        final_models[name] = model
        test_predictions = model.predict(X_test)
        metrics = calculate_metrics(y_test, test_predictions)
        final_rows.append({"Model": name, **{f"Test_{k}": v for k, v in metrics.items()}})

    results = validation_results.merge(pd.DataFrame(final_rows), on="Model")
    results.to_csv(RESULTS_PATH, index=False)
    print("\nValidation and test results:")
    print(results.to_string(index=False))

    best_model = final_models[best_name]
    best_predictions = best_model.predict(X_test)
    joblib.dump(best_model, MODEL_PATH)
    save_model_plots(best_name, y_test, best_predictions, final_models["Random Forest"])

    print(f"\nSaved selected model to: {MODEL_PATH.name}")
    print(f"Saved model comparison to: {RESULTS_PATH.name}")
    print(f"Saved plots to: {PLOTS_DIR}")
    print("Project run completed successfully.")


if __name__ == "__main__":
    main()

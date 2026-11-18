# House Price Prediction using Machine Learning

A regression project that explores housing data and compares Linear Regression with Random Forest Regressor to estimate house prices from property features.

> **Dataset note:** The supplied CSV contains 545 rows and 13 columns. Its currency and geographic coverage are not established by the CSV itself. Until the original dataset source is confirmed, this project labels predictions as **dataset price units** and does not describe them as INR or Mumbai property prices.

## Project objectives

- Explore and visualize housing data.
- Prepare numerical and categorical features for machine learning.
- Compare Linear Regression and Random Forest Regressor.
- Evaluate models using R², MAE, and RMSE.
- Save the selected fitted pipeline and use it to predict a new example.

## Dataset

File: `Housing.csv`

**Dataset availability:** `Housing.csv` is required to run the project locally but is not included in this GitHub repository. Obtain the dataset from its original source after checking its license and usage permissions, then place it in the project root folder.

- 545 records and 13 columns in the supplied file.
- Target: `price`.
- Predictors: `area`, `bedrooms`, `bathrooms`, `stories`, `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `parking`, `prefarea`, and `furnishingstatus`.

The dataset source/license should be confirmed and credited here before public redistribution. Do not claim a location or currency unless supported by the original source.

## Methods

1. Load and inspect the CSV; check missing values and duplicates.
2. Create exploratory plots for price distribution, area vs. price, and numeric correlations.
3. Create two deterministic derived features: `bed_bath_count` and `area_per_bed_bath`.
4. Split data into training, validation, and test sets (60% / 20% / 20%).
5. Use scikit-learn Pipelines and ColumnTransformer for imputation, categorical one-hot encoding, and numeric scaling for Linear Regression.
6. Compare Linear Regression and Random Forest using validation RMSE to select a model.
7. Refit models on train + validation data and report final test metrics.
8. Save the selected complete pipeline as `house_price_model.joblib`.

Preprocessing is fitted within each training pipeline to reduce data leakage. The validation set is used for model selection; the test set is kept for final evaluation.

## Project structure

```text
House_Price_Prediction_GitHub/
├── Housing.csv               # Local dataset (not uploaded to GitHub)
├── house_price_prediction.py
├── preprocessing.py
├── predict_house.py
├── requirements.txt
├── README.md
├── .gitignore
├── model_results.csv       # generated after training
├── house_price_model.joblib # generated after training
└── plots/                  # generated after training
```

## Setup and run (VS Code / terminal)

Create and activate a virtual environment (recommended), then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python house_price_prediction.py
```

Then run the example prediction:

```powershell
python predict_house.py
```

On macOS/Linux, activate with `source .venv/bin/activate` and run the same Python commands.

## Outputs

Running the training script creates:

- `model_results.csv` — validation and test metrics for both models.
- `house_price_model.joblib` — selected fitted preprocessing + regression pipeline.
- `plots/price_distribution.png`
- `plots/area_vs_price.png`
- `plots/correlation_heatmap.png`
- `plots/actual_vs_predicted.png`
- `plots/feature_importance.png` — Random Forest feature importance for interpretation, not a claim that Random Forest is the best predictor.

## Metrics

- **R²:** proportion of target variance explained by the model (higher is generally better).
- **MAE:** average absolute prediction error (lower is better).
- **RMSE:** square root of mean squared error; penalizes larger errors more strongly (lower is better).

Results from the current run (your results may vary slightly with software versions):

| Model | Validation R² | Validation MAE | Validation RMSE | Test R² | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.6681 | 728,957 | 988,561 | 0.5941 | 1,059,804 | 1,432,327 |
| Linear Regression | 0.6487 | 759,742 | 1,016,995 | 0.6583 | 966,320 | 1,314,226 |

The current script selects the model with the lower **validation RMSE**; in this run that is Random Forest. On the separate test set, Linear Regression scores better. This difference is worth discussing: model selection on a small validation sample can be uncertain, and the final test set should not be used to tune the model. `model_results.csv` contains the full metrics.

## Limitations

- This is a learning project, not a production property valuation tool.
- Model performance depends on the supplied dataset and its representativeness.
- Currency, location, data collection period, and dataset license must be verified from the original source.
- The example in `predict_house.py` is illustrative; replace it with valid property details.

## Author

Sayera Shaikh

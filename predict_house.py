"""Make a prediction for a new house using the saved model.

Run the main training script first, then run:
    python predict_house.py
"""

from pathlib import Path
import joblib
import pandas as pd

from preprocessing import prepare_features

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.joblib"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run house_price_prediction.py first.")

    model = joblib.load(MODEL_PATH)

    # Example input only. Replace these values with the house details you want.
    new_house = pd.DataFrame([{
        "area": 5000,
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "mainroad": "yes",
        "guestroom": "no",
        "basement": "no",
        "hotwaterheating": "no",
        "airconditioning": "yes",
        "parking": 1,
        "prefarea": "yes",
        "furnishingstatus": "semi-furnished",
    }])

    model_input = prepare_features(new_house)
    predicted_price = model.predict(model_input)[0]
    print(f"Predicted house price (dataset units): {predicted_price:,.2f}")
    print("Note: Confirm the dataset's currency and geographic coverage before interpreting this value.")


if __name__ == "__main__":
    main()

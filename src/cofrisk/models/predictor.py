from pathlib import Path

import joblib
import pandas as pd

from cofrisk.data.validation import EXPECTED_FEATURES


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "cofrisk_xgb_pipeline.joblib"
)

THRESHOLD = 0.20


class RiskPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, features: dict) -> dict:
        df = pd.DataFrame([features], columns=EXPECTED_FEATURES)

        probability = float(
            self.model.predict_proba(df)[0, 1]
        )

        risk_class = int(probability >= THRESHOLD)

        return {
            "default_probability": probability,
            "risk_class": risk_class,
        }
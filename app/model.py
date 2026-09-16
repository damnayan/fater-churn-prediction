import os
import numpy as np
import xgboost as xgb
from app.schemas import UserFeatures, PredictionResponse

MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_churn_model.json")


class ChurnModel:

    def __init__(self, model_path: str = MODEL_PATH):
        self.model = xgb.XGBClassifier()
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at target path: {model_path}"
            )
        self.model.load_model(model_path)

    @staticmethod
    def classify_segment(is_churn: int, total_logins: int, prob: float) -> str:
        """Categorizes user into Loyal, Medium Risk, or High Risk based on business rules."""
        if is_churn == 1 or prob >= 0.5:
            return "High Risk" if total_logins == 0 else "Medium Risk"
        return "Loyal"

    def predict_single(self, features: UserFeatures) -> PredictionResponse:
        row = np.array(
            [
                [
                    features.baby_age_months,
                    features.total_points,
                    features.total_codes,
                    features.total_logins,
                    features.total_missions,
                    features.tenure_days,
                ]
            ]
        )

        proba = float(self.model.predict_proba(row)[0, 1])
        is_churn = int(proba >= 0.5)
        segment = self.classify_segment(is_churn, features.total_logins, proba)

        return PredictionResponse(
            churn_probability=round(proba, 4),
            is_churn=is_churn,
            risk_segment=segment,
        )


churn_service = None


def get_model() -> ChurnModel:
    global churn_service
    if churn_service is None:
        churn_service = ChurnModel()
    return churn_service

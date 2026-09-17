import os
import pandas as pd
from xgboost import XGBClassifier

# 1. Load prepared master dataset
df = pd.read_csv("fater_final_results.csv")

# 2. Select model features & target
features = [
    "ETA_MM_BambinoTODAY",
    "total_points",
    "total_codes",
    "total_logins",
    "total_missions",
    "tenure_days",
]

X = df[features]
y = df["is_churn"]

# 3. Fit XGBoost classifier
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    eval_metric="logloss",
)
model.fit(X, y)

# 4. Save model artifact
os.makedirs("models", exist_ok=True)
model.save_model("models/xgb_churn_model.json")
print("Original trained model successfully exported to models/xgb_churn_model.json")

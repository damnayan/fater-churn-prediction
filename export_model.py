import os
import pandas as pd
from xgboost import XGBClassifier

# 1. Load original datasets
utenti = pd.read_csv("utenti_reduced.csv")
accessi = pd.read_csv("accessi_reduced.csv")
codici = pd.read_csv("codici_reduced.csv")
missions = pd.read_csv("missioni_reduced.csv")
rewusers = pd.read_csv("rewusers_reduced.csv")


# 2. Clean identifiers
def clean_id(df, col):
    df[col] = df[col].astype(str).str.strip().str.lower()


for d, c in [
    (utenti, "idSSO"),
    (accessi, "idsso"),
    (codici, "userId"),
    (missions, "userId"),
    (rewusers, "idSSO"),
    (rewusers, "userId"),
]:
    clean_id(d, c)

# 3. Aggregate user actions
user_map = utenti[["idSSO"]].merge(
    rewusers[["idSSO", "userId"]], on="idSSO", how="left"
)

codici_agg = (
    codici.groupby("userId")
    .agg(total_points=("points", "sum"), total_codes=("code", "count"))
    .reset_index()
)

accessi_agg = (
    accessi.groupby("idsso")
    .agg(
        total_logins=("created_at", "count"),
        last_login_date=("created_at", "max"),
    )
    .reset_index()
)

missions_agg = (
    missions.groupby("userId")
    .agg(total_missions=("missionDetailId", "count"))
    .reset_index()
)

# 4. Master merge
df_final = user_map.copy()
df_final = df_final.merge(codici_agg, on="userId", how="left")
df_final = df_final.merge(missions_agg, on="userId", how="left")
df_final = df_final.merge(utenti, on="idSSO", how="left")
df_final = df_final.merge(
    accessi_agg, left_on="idSSO", right_on="idsso", how="left"
)

behavior_cols = [
    "total_points",
    "total_codes",
    "total_logins",
    "total_missions",
]
df_final[behavior_cols] = df_final[behavior_cols].fillna(0)

# 5. Target creation (90 days inactivity threshold)
df_final["last_login_date"] = pd.to_datetime(df_final["last_login_date"])
df_final["DtaRegUserData"] = pd.to_datetime(df_final["DtaRegUserData"])
snapshot_date = df_final["last_login_date"].max()

df_final["days_since_last_login"] = (
    snapshot_date - df_final["last_login_date"]
).dt.days.fillna(999)
df_final["is_churn"] = (df_final["days_since_last_login"] > 90).astype(int)

df_final["tenure_days"] = (snapshot_date - df_final["DtaRegUserData"]).dt.days
df_final["tenure_days"] = df_final["tenure_days"].fillna(
    df_final["tenure_days"].median()
)
df_final["ETA_MM_BambinoTODAY"] = df_final["ETA_MM_BambinoTODAY"].fillna(
    df_final["ETA_MM_BambinoTODAY"].median()
)

# 6. Fit XGBoost on full feature set
features = [
    "ETA_MM_BambinoTODAY",
    "total_points",
    "total_codes",
    "total_logins",
    "total_missions",
    "tenure_days",
]

X = df_final[features]
y = df_final["is_churn"]

model = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
)
model.fit(X, y)

# 7. Save production artifact
os.makedirs("models", exist_ok=True)
model.save_model("models/xgb_churn_model.json")
print("Original trained model successfully exported to models/xgb_churn_model.json")

# ==========================================
# QuakeGuard v3 - Model Training Module
# ==========================================
# Run this script ONCE to retrain all models.
# Output: damage_model.pkl, casualty_model.pkl,
#         blackout_model.pkl, triage_model.pkl, encoders.pkl
#
# Model choices:
#   Damage    → GradientBoostingClassifier  (handles non-linear score thresholds well)
#   Casualty  → RandomForestClassifier      (robust to noisy population features)
#   Blackout  → GradientBoostingClassifier  (magnitude + infra interaction is non-linear)
#   Triage    → RandomForestClassifier      (4-feature dataset, avoids overfitting)
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

eq  = pd.read_csv("earthquake_data.csv")
pat = pd.read_csv("patient_data.csv")
print("✅ Data loaded")

# Encode labels
encoders = {}
def encode(df, col):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    return df

for c in ["damage", "casualty_risk", "blackout"]:
    eq = encode(eq, c)
pat = encode(pat, "triage")
joblib.dump(encoders, "encoders.pkl")

# Features
EQ_FEATS  = ["magnitude","depth","population_density","infrastructure","urban","time_night","distance"]
PAT_FEATS = ["age","injury_severity","conscious","heart_rate"]

X = eq[EQ_FEATS]
Xp = pat[PAT_FEATS]

Xtr, Xte, ydt, yde = train_test_split(X, eq["damage"],       test_size=0.2, random_state=42)
_,   _,   yct, yce = train_test_split(X, eq["casualty_risk"],test_size=0.2, random_state=42)
_,   _,   ybt, ybe = train_test_split(X, eq["blackout"],      test_size=0.2, random_state=42)
Xpt, Xpe, ytt, yte = train_test_split(Xp,pat["triage"],       test_size=0.2, random_state=42)

print("🚀 Training...")
damage_model   = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)
casualty_model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
blackout_model = GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)
triage_model   = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)

damage_model.fit(Xtr, ydt)
casualty_model.fit(Xtr, yct)
blackout_model.fit(Xtr, ybt)
triage_model.fit(Xpt, ytt)
print("✅ Training done")

for name, model, Xte_, yte_ in [
    ("Damage",   damage_model,   Xte, yde),
    ("Casualty", casualty_model, Xte, yce),
    ("Blackout", blackout_model, Xte, ybe),
    ("Triage",   triage_model,   Xpe, yte),
]:
    print(f"\n📊 {name}: acc={accuracy_score(yte_, model.predict(Xte_)):.3f}")
    print(classification_report(yte_, model.predict(Xte_)))

joblib.dump(damage_model,   "damage_model.pkl")
joblib.dump(casualty_model, "casualty_model.pkl")
joblib.dump(blackout_model, "blackout_model.pkl")
joblib.dump(triage_model,   "triage_model.pkl")
print("\n💾 All models saved.")

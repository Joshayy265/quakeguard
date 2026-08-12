# ==========================================
# QuakeGuard v3 - Data Simulation Module
# ==========================================
# Generates:
#   earthquake_data.csv  — 1500 earthquake events with 7 features + 3 labels
#   patient_data.csv     — 300 patient records with 4 features + 1 triage label
# ==========================================

import numpy as np
import pandas as pd

np.random.seed(42)

FIRST_NAMES = [
    "Arjun","Priya","Ravi","Sunita","Vikram","Meena","Aditya","Kavya",
    "Rahul","Ananya","Sanjay","Deepa","Karthik","Lakshmi","Nikhil","Pooja",
    "Suresh","Divya","Manoj","Rekha","Amit","Shreya","Dinesh","Usha",
    "Rohit","Swathi","Mohan","Radha","Venkat","Geetha","Harish","Nandini",
]
LAST_NAMES = [
    "Sharma","Patel","Reddy","Nair","Iyer","Rao","Kumar","Singh",
    "Verma","Pillai","Mehta","Joshi","Gupta","Bhat","Chaudhary","Desai",
]
INJURY_TYPES = {
    1: "Lacerations / bruises",
    2: "Fracture / internal bleeding",
    3: "Crush injury / trauma",
}
BLOOD_GROUPS = ["A+","A-","B+","B-","O+","O-","AB+","AB-"]
CONDITIONS    = ["None","Diabetes","Hypertension","Asthma","Heart disease"]
OCCUPATIONS   = ["Student","Teacher","Farmer","Engineer","Doctor","Shopkeeper",
                 "Construction worker","Homemaker","Government employee","Driver"]


def generate_earthquake_data(n=1500):
    data = pd.DataFrame({
        "magnitude":          np.round(np.random.uniform(4.0, 8.5, n), 1),
        "depth":              np.round(np.random.uniform(5, 100, n), 1),
        "population_density": np.random.randint(50, 10000, n),
        "infrastructure":     np.random.randint(1, 11, n),
        "urban":              np.random.choice([0, 1], n),
        "time_night":         np.random.choice([0, 1], n),
        "distance":           np.round(np.random.uniform(1, 300, n), 1),
    })

    def damage_label(row):
        score = (
            row["magnitude"] * 1.5
            + row["population_density"] / 3000
            - row["depth"] / 60
            - row["infrastructure"] / 1.5
            + np.random.normal(0, 0.7)
        )
        return "High" if score > 9 else ("Medium" if score > 6 else "Low")

    def casualty_label(row):
        score = (
            row["population_density"] / 2000
            + (10 - row["infrastructure"]) / 2.5
            + row["time_night"] * 1.5
            + np.random.normal(0, 0.6)
        )
        return "High" if score > 7 else ("Medium" if score > 4 else "Low")

    def blackout_label(row):
        score = (
            row["magnitude"]
            + (10 - row["infrastructure"])
            + row["distance"] / 120
            + np.random.normal(0, 0.4)
        )
        return "Failure" if score > 14 else ("Risk" if score > 9 else "Safe")

    data["damage"]       = data.apply(damage_label, axis=1)
    data["casualty_risk"] = data.apply(casualty_label, axis=1)
    data["blackout"]     = data.apply(blackout_label, axis=1)
    return data


def generate_patient_data(n=300):
    patients = pd.DataFrame({
        "age":              np.random.randint(1, 90, n),
        "injury_severity":  np.random.choice([1, 2, 3], n),
        "conscious":        np.random.choice([0, 1], n),
        "heart_rate":       np.random.randint(50, 150, n),
    })

    def triage_label(row):
        score = (
            row["injury_severity"] * 1.5
            + (1 - row["conscious"]) * 2.5
            + (row["heart_rate"] > 120) * 1.0
            + np.random.normal(0, 0.3)
        )
        return "Critical" if score > 5 else ("Serious" if score > 3 else "Minor")

    patients["triage"] = patients.apply(triage_label, axis=1)
    return patients


def generate_rich_patients(lat, lon, n=20, seed=None):
    """Generate patients with full demographic profiles for map display."""
    if seed is not None:
        np.random.seed(seed)
    rng = np.random

    ages              = rng.randint(1, 88, n)
    injury_severities = rng.choice([1, 2, 3], n, p=[0.4, 0.35, 0.25])
    conscious         = rng.choice([0, 1], n, p=[0.3, 0.7])
    heart_rates       = rng.randint(55, 148, n)
    blood_pressure_s  = rng.randint(80, 180, n)
    blood_pressure_d  = rng.randint(50, 110, n)
    spo2              = rng.randint(82, 100, n)

    first = rng.choice(FIRST_NAMES, n)
    last  = rng.choice(LAST_NAMES, n)
    names = [f"{f} {l}" for f, l in zip(first, last)]

    patients = pd.DataFrame({
        "name":             names,
        "age":              ages,
        "gender":           rng.choice(["Male", "Female"], n),
        "blood_group":      rng.choice(BLOOD_GROUPS, n),
        "occupation":       rng.choice(OCCUPATIONS, n),
        "pre_condition":    rng.choice(CONDITIONS, n, p=[0.55, 0.12, 0.15, 0.08, 0.10]),
        "injury_severity":  injury_severities,
        "injury_type":      [INJURY_TYPES[s] for s in injury_severities],
        "conscious":        conscious,
        "heart_rate":       heart_rates,
        "bp_systolic":      blood_pressure_s,
        "bp_diastolic":     blood_pressure_d,
        "spo2":             spo2,
        "lat":              lat + rng.uniform(-0.35, 0.35, n),
        "lon":              lon + rng.uniform(-0.35, 0.35, n),
        "rescue_status":    rng.choice(
            ["Awaiting rescue", "Rescue in progress", "Evacuated to hospital", "Field treated"],
            n, p=[0.35, 0.30, 0.20, 0.15]
        ),
    })
    return patients


if __name__ == "__main__":
    eq  = generate_earthquake_data(1500)
    pat = generate_patient_data(300)
    eq.to_csv("earthquake_data.csv", index=False)
    pat.to_csv("patient_data.csv", index=False)
    print("✅ Data simulation complete")
    print(eq["damage"].value_counts())
    print(pat["triage"].value_counts())

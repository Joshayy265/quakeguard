# 🌍 QuakeGuard — Earthquake Response Dashboard

Given an earthquake's parameters, four machine-learning models answer four
different questions at once: how badly will structures be damaged, what is the
casualty risk, will communications go down, and which of the patients on scene
should be treated first?

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **📖 [Full technical documentation →](docs/TECHNICAL.md)** — 800 lines covering
> every model, panel and design decision in detail.

---

## The idea

Disaster response is a triage problem at two scales at once. At the macro scale
you are deciding where to send resources; at the micro scale you are deciding
which patient gets the last ambulance. QuakeGuard models both, which is why it
is four models rather than one:

| Model | Predicts | Algorithm | Why this one |
|---|---|---|---|
| **Damage** | Structural severity | GradientBoosting | Magnitude→damage is sharply non-linear with threshold effects; boosting captures those step changes |
| **Casualty** | Human casualty risk | RandomForest | Robust to the outliers that dominate casualty data — a few catastrophic events skew everything |
| **Blackout** | Comms/power status | GradientBoosting | Depends on interacting infrastructure features rather than any single one |
| **Triage** | Per-patient priority | RandomForest | Multi-class over vitals; feature importances stay interpretable, which matters for a medical decision |

Splitting the problem this way means each model trains on the features that
actually matter to it, instead of one model diluting itself across four
unrelated targets.

---

## Quick start

```bash
git clone https://github.com/Joshayy265/quakeguard.git
cd quakeguard
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Set the earthquake parameters in the sidebar and the
whole dashboard recomputes.

> **Note**: The committed `.pkl` models were trained with **scikit-learn 1.6.1**,
> which `requirements.txt` pins exactly — any other version raises
> `InconsistentVersionWarning` and may return invalid results. To move to a newer
> version, unpin it and run `python models.py` to retrain.

---

## What's in the dashboard

- **KPI strip** — damage severity, casualty risk, blackout status, patients by priority
- **Resource allocation** — ambulances, rescue teams and shelters derived from the predictions
- **Explainability panel** — which features drove each prediction, and by how much
- **Interactive map** — folium map of affected buildings and patient locations
- **Historical comparison** — the current scenario against real recorded earthquakes
- **Simulation mode** — step time forward and watch the situation evolve

---

## Project structure

```
quakeguard/
├── app.py                  # Streamlit dashboard — all UI and logic
├── models.py               # Trains and saves all four models
├── data_simulation.py      # Synthetic data generation
├── earthquake_data.csv     # 1,500-row earthquake training set
├── patient_data.csv        # 300-row patient training set
├── *.pkl                   # Four trained models + label encoders
├── docs/TECHNICAL.md       # Full documentation
├── LICENSE
└── README.md
```

---

## Limitations

All training data is **synthetic**. Real earthquake damage depends on soil
liquefaction, building codes, aftershock sequences and response capacity — none
of which are modelled here. Patient vitals are generated once and never
deteriorate, and resources are allocated but not routed. This is a demonstration
of an end-to-end ML system, not a tool for real emergency planning.

The [technical documentation](docs/TECHNICAL.md#11-limitations--future-work) goes
through these in detail.

---

## License

MIT — see [LICENSE](LICENSE).

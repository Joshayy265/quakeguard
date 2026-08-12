# 🌍 QuakeGuard — Complete Technical Documentation

> This is the deep-dive reference. For a project overview and quick start, see
> the [README](../README.md).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Quick Start](#2-quick-start)
3. [Architecture Overview](#3-architecture-overview)
4. [Data Layer — data_simulation.py](#4-data-layer)
5. [Machine Learning Layer — models.py](#5-machine-learning-layer)
   - 5.1 Why these specific models?
   - 5.2 Damage Model (GradientBoosting)
   - 5.3 Casualty Model (RandomForest)
   - 5.4 Blackout Model (GradientBoosting)
   - 5.5 Triage Model (RandomForest)
   - 5.6 Label Encoding
   - 5.7 Train/Test Split
   - 5.8 Evaluation
6. [Application Layer — app.py](#6-application-layer)
   - 6.1 Page Configuration & CSS Theme
   - 6.2 Model Loading (Caching)
   - 6.3 Session State
   - 6.4 Patient Generation (Rich Profiles)
   - 6.5 Building Damage Simulation
   - 6.6 Resource Allocation Formula
   - 6.7 Explainability Engine
   - 6.8 Historical Simulation
   - 6.9 Map System
   - 6.10 Simulation Mode
   - 6.11 KPI Strip
   - 6.12 Alert System
7. [Every Dashboard Panel Explained](#7-every-dashboard-panel-explained)
8. [Demo Guide — How to Present This](#8-demo-guide)
9. [Design Decisions FAQ](#9-design-decisions-faq)
10. [File Structure](#10-file-structure)
11. [Limitations & Future Work](#11-limitations--future-work)

---

## 1. Project Overview

QuakeGuard is a **machine-learning powered earthquake response simulation dashboard** built in Python/Streamlit. It was designed as an end-to-end ML project that demonstrates:

- Synthetic data generation that mimics real geophysical and medical relationships
- Training of four separate classification models on different sub-problems
- A real-time interactive dashboard that takes user inputs, runs all four models, and presents results, explanations, and derived recommendations

**The core question being answered**: Given the parameters of an earthquake (magnitude, depth, location, infrastructure quality, etc.) what is the likely: (a) physical damage severity, (b) human casualty risk, (c) communication system status, and (d) medical priority of each patient found at the scene?

---

## 2. Quick Start

```bash
# Step 1: Clone
git clone https://github.com/Joshayy265/quakeguard.git
cd quakeguard

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3 (optional): Retrain upgraded models
python models.py

# Step 4: Launch
streamlit run app.py
```

> **Note**: The `.pkl` files committed to this repo were trained with **scikit-learn 1.6.1**,
> which `requirements.txt` now pins exactly. On any other version sklearn raises
> `InconsistentVersionWarning` and may return invalid results.
> To use a newer scikit-learn, unpin it and run `python models.py` to retrain against your installed version.

---

## 3. Architecture Overview

```
User Input (Streamlit Sidebar)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                  INPUT VECTOR                        │
│  [magnitude, depth, population_density,              │
│   infrastructure, urban, time_night, distance]       │
└──────────┬──────────┬──────────┬────────────────────┘
           │          │          │
           ▼          ▼          ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Damage   │ │ Casualty │ │ Blackout │
    │ GB Model │ │ RF Model │ │ GB Model │
    └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │
         ▼            ▼            ▼
      High/Med/   High/Med/   Failure/Risk/
        Low         Low          Safe

Patient Vitals ──► Triage RF Model ──► Critical/Serious/Minor

Damage + Casualty + Blackout + Triage Counts
         │
         ▼
  Resource Allocation Formula ──► Ambulances, Rescue Teams,
                                   Med Kits, Helicopters, etc.
         │
         ▼
    Explainability Engine ──► "Why this prediction?"

    Map Builder ──► Folium Dark Map with patient/building markers
```

---

## 4. Data Layer

**File**: `data_simulation.py`

### Why synthetic data?

Real earthquake response datasets are fragmented, inconsistently labeled, and not publicly available at the granularity needed (per-event damage + per-patient triage). Synthetic data lets us define exactly what relationships we want the models to learn, producing clean, balanced, reproducible training sets.

### `generate_earthquake_data(n=1500)`

Creates 1500 earthquake event records. Each row represents one earthquake event with:

| Feature | Type | Range | Meaning |
|---------|------|-------|---------|
| `magnitude` | float | 4.0–8.5 | Richter scale energy release |
| `depth` | float | 5–100 km | Focal depth below surface |
| `population_density` | int | 50–10,000 /km² | People in affected area |
| `infrastructure` | int | 1–10 | Quality of buildings/roads (10=best) |
| `urban` | binary | 0/1 | Rural vs Urban |
| `time_night` | binary | 0/1 | Day vs Night |
| `distance` | float | 1–300 km | Distance of observer from epicenter |

**Why 1500 rows?** Enough to give each of the three output classes (Low/Medium/High) ~300–600 samples with balanced training, while keeping training fast. More rows would improve accuracy marginally but slow the demo.

#### Label Generation Logic

The labels are not random — they are computed from physics-inspired score functions:

**Damage severity score:**
```python
score = magnitude * 1.5          # magnitude has the biggest effect on shaking
      + population_density / 3000 # denser areas = more structural exposure
      - depth / 60                # deeper = less surface energy
      - infrastructure / 1.5     # better buildings resist damage
      + noise(μ=0, σ=0.7)        # real-world variability

if score > 9  → "High"
if score > 6  → "Medium"
else          → "Low"
```

The thresholds (9 and 6) were chosen by running the formula on the full range and ensuring roughly equal class distribution (avoiding heavily imbalanced classes which would cause the model to learn the majority class only).

**Casualty risk score:**
```python
score = population_density / 2000  # more people = more at risk
      + (10 - infrastructure) / 2.5 # weak infra = more injuries
      + time_night * 1.5            # sleeping people evacuate slower
      + noise(μ=0, σ=0.6)

if score > 7  → "High"
if score > 4  → "Medium"
else          → "Low"
```

Note: Magnitude does NOT appear in casualty score — this is intentional. Casualties depend more on how many people are in the area and whether buildings protect them, not purely on magnitude.

**Communication blackout score:**
```python
score = magnitude                       # shaking destroys equipment
      + (10 - infrastructure)           # old/weak towers fail faster
      + distance / 120                  # remote areas have less redundancy
      + noise(μ=0, σ=0.4)

if score > 14 → "Failure"
if score > 9  → "Risk"
else          → "Safe"
```

**The noise term** (`np.random.normal(0, σ)`) is critical. Without it, the score function would be perfectly deterministic and the model would achieve near 100% accuracy by memorizing the exact boundaries — it would learn the formula, not the underlying pattern. The noise forces the model to learn probabilistic boundaries, making it more realistic and generalizable.

### `generate_patient_data(n=300)`

Creates 300 patient records representing earthquake casualties. Each row is one patient.

| Feature | Type | Range | Meaning |
|---------|------|-------|---------|
| `age` | int | 1–88 | Patient age in years |
| `injury_severity` | int | 1, 2, 3 | 1=Lacerations, 2=Fracture, 3=Crush injury |
| `conscious` | binary | 0/1 | Whether patient is conscious |
| `heart_rate` | int | 55–148 bpm | Measured pulse |

**Triage label logic:**
```python
score = injury_severity * 1.5      # injury type is the strongest predictor
      + (1 - conscious) * 2.5      # unconscious patients are almost always critical
      + (heart_rate > 120) * 1.0   # tachycardia indicates shock/distress
      + noise(μ=0, σ=0.3)

if score > 5 → "Critical"
if score > 3 → "Serious"
else         → "Minor"
```

**Why these specific thresholds?**
- An unconscious patient (conscious=0) contributes +2.5 to the score alone — nearly enough for "Serious" classification by itself. This matches real triage practice (START triage, SALT triage) where unconsciousness is a near-automatic critical flag.
- Injury severity 3 (crush injury) × 1.5 = 4.5, already near "Serious" without any other factors.
- Heart rate >120 (tachycardia) adds +1.0, a reasonable modifier rather than a primary signal.

### `generate_rich_patients()` (used at runtime in app.py)

Unlike the training dataset, the runtime patients have full demographic profiles:
- **Name** (realistic Indian names using random first + last name lists)
- **Gender, blood group, occupation, pre-existing conditions**
- **Vital signs**: heart rate, BP (systolic/diastolic), SpO₂
- **Injury type** (text description based on severity)
- **Rescue status** (Awaiting rescue / In progress / Evacuated / Field treated)
- **GPS coordinates** (scattered around the epicenter within ±0.30°)

These rich profiles make the patient markers on the map meaningful — clicking any red/orange/green dot shows you exactly who the person is, their medical situation, and what's being done for them.

---

## 5. Machine Learning Layer

**File**: `models.py`

### 5.1 Why These Specific Models?

A critical question. We have four sub-problems — why not use the same model for all four?

#### Why NOT a single model?

Each classification target has different characteristics:
- **Damage** has a strong non-linear interaction between magnitude and infrastructure (high magnitude × weak infrastructure → exponentially worse outcome). Tree-based ensembles handle this well.
- **Casualty** risk depends more on the combination of population + time + infrastructure (somewhat more additive), which Random Forest handles well due to its averaging effect.
- **Blackout** is dominated by 2 features (magnitude + infrastructure), a situation where GradientBoosting's sequential error correction shines.
- **Triage** has only 4 features with clear clinical rules (unconscious → critical). RandomForest is appropriate here to avoid overfitting on such a small feature set.

#### Why NOT Logistic Regression?

The original v1 used Logistic Regression for Casualty and Blackout. The problem: Logistic Regression assumes **linear decision boundaries**. But the relationship between, say, magnitude and damage is multiplicative — a M7.0 quake with infrastructure=2 is far more catastrophic than the linear sum would predict. Logistic Regression would underfit these interactions.

#### Why NOT a Neural Network?

With 1500 training samples and only 7 features, a neural network would be massive overkill and would almost certainly overfit. Tree-based methods outperform neural networks on tabular data at this scale (this is well-established in the ML literature — see "Why do tree-based models still outperform deep learning on tabular data?" Grinsztajn et al., 2022). Neural networks need tens of thousands of samples to regularize properly.

#### Why NOT a single Decision Tree?

Decision Trees (used in v1) are unstable — small changes in the input data can completely change the tree structure. They also overfit on training data without extensive pruning. Ensemble methods (Random Forest = many trees, Gradient Boosting = sequential trees) are strictly better for prediction quality.

---

### 5.2 Damage Severity Model

**Algorithm**: `GradientBoostingClassifier`
**Target**: Low / Medium / High damage
**Features**: All 7 earthquake features
**Hyperparameters**: `n_estimators=150, max_depth=4, learning_rate=0.08`

**How Gradient Boosting works**:
1. Start with a naive prediction (e.g., always predict the majority class)
2. Build a shallow decision tree on the **residual errors** from step 1
3. Add this tree to the ensemble with weight = learning_rate × tree_output
4. Repeat 150 times, each tree correcting what previous trees got wrong
5. Final prediction = sum of all 150 tree contributions

**Why max_depth=4?** Each tree can ask at most 4 questions before making a prediction. This prevents any single tree from memorizing training data while still capturing 2nd and 3rd order feature interactions (e.g., "if magnitude > 7 AND infrastructure < 4 AND depth < 20...").

**Why learning_rate=0.08?** Lower learning rates require more trees but produce more robust models. This is a classic bias-variance trade-off: small steps toward the correct answer → less risk of overshooting.

**Feature importance** (approximate, from training):
- `magnitude`: ~42% importance — the primary driver of physical damage
- `infrastructure`: ~41% — almost equally important; weak buildings amplify shaking
- `population_density`: ~15% — affects exposure, not damage per se
- `depth`, `distance`, `urban`, `time_night`: <5% combined for this target

---

### 5.3 Casualty Risk Model

**Algorithm**: `RandomForestClassifier`
**Target**: Low / Medium / High casualty risk
**Features**: All 7 earthquake features
**Hyperparameters**: `n_estimators=150, max_depth=8`

**How Random Forest works**:
1. Draw 150 bootstrap samples of the training data (random sampling with replacement)
2. Train a deep decision tree on each bootstrap sample, but at each split point, only consider a random subset of features (typically √7 ≈ 2-3 features)
3. Final prediction = majority vote across all 150 trees

**Why RandomForest for casualty risk specifically?**
Casualty risk depends on population density (continuous, high variance), time of day (binary, can flip predictions), and infrastructure (ordinal). The combination of these three factors is relatively additive but with outlier noise from the population density feature (which ranges 50–10,000 — a 200× range). Random Forest is resistant to outliers because averaging many trees smooths out extreme values.

**Why max_depth=8?** Deeper than the damage model because the feature space is more complex — we need to distinguish "low population + bad infrastructure at night" from "high population + good infrastructure during day." The extra depth captures these 3-way interactions.

---

### 5.4 Blackout Model

**Algorithm**: `GradientBoostingClassifier` (same architecture as Damage model)
**Target**: Safe / Risk / Failure
**Features**: All 7 earthquake features
**Hyperparameters**: `n_estimators=150, max_depth=4, learning_rate=0.08`

**Why Gradient Boosting here again?**
Looking at the score formula, communication blackout is primarily driven by magnitude + infrastructure, with distance as a secondary factor. This creates a sharp threshold effect: once magnitude + (10-infra) exceeds ~14, failure is nearly certain. GradientBoosting is excellent at learning sharp boundaries because each successive tree focuses on the hard cases near the boundary.

---

### 5.5 Triage Model

**Algorithm**: `RandomForestClassifier`
**Target**: Minor / Serious / Critical
**Features**: `age, injury_severity, conscious, heart_rate`
**Hyperparameters**: `n_estimators=150, max_depth=6`

**Why RandomForest with only 4 features?**
With only 4 features, a single Decision Tree could memorize the training data completely (overfitting). Random Forest's feature subsampling (√4 = 2 features per split) forces each tree to learn different aspects of the data, and averaging their predictions gives a more robust model.

**Why NOT Logistic Regression for triage?**
Because `conscious` (binary, huge impact) and `injury_severity` (ordinal, non-linear impact) interact non-linearly with the outcome. An unconscious patient with severity=1 (minor lacerations) might still be Serious, but a conscious patient with severity=3 might only be Serious rather than Critical if their heart rate is normal. These conditional interactions require a tree-based model to capture correctly.

**Feature importance** (triage model):
- `injury_severity`: ~50% — the primary clinical signal
- `conscious`: ~27% — unconsciousness is a strong critical indicator
- `heart_rate`: ~18% — tachycardia confirms physiological distress
- `age`: ~5% — minor effect; elderly patients slightly more vulnerable

---

### 5.6 Label Encoding

All target labels (strings like "High", "Low", "Critical") must be converted to integers for scikit-learn models. `LabelEncoder` from sklearn handles this alphabetically:

```
damage:        High=0, Low=1, Medium=2
casualty_risk: High=0, Low=1, Medium=2
blackout:      Failure=0, Risk=1, Safe=2
triage:        Critical=0, Minor=1, Serious=2
```

The `encoders.pkl` file stores all four `LabelEncoder` objects. At prediction time, `inverse_transform()` converts the integer prediction back to a human-readable string.

**Why store encoders separately from models?**
The models predict integers internally. To display "High" instead of "0" to the user, you need the same encoder that was used during training. If you retrain models but forget to save the new encoders, predictions will be wrong (silently, which is dangerous). Keeping them together in one file ensures consistency.

---

### 5.7 Train/Test Split

```python
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
```

- **80/20 split**: 1200 training, 300 testing samples
- **random_state=42**: Ensures reproducibility — every run produces the same split
- All four earthquake targets share the same `X_train`/`X_test` split (important: the feature matrix is the same, only the target vectors differ). This prevents any accidental data leakage.

---

### 5.8 Evaluation

After training, each model is evaluated on the held-out test set using:
- **Accuracy score**: Overall fraction of correct predictions
- **Classification report**: Per-class precision, recall, F1-score
  - **Precision**: Of all times I predicted "High", how often was I right?
  - **Recall**: Of all actual "High" events, how many did I catch?
  - **F1**: Harmonic mean of precision and recall (balanced measure)

Expected accuracy ranges (on this synthetic dataset):
- Damage model: ~82–88%
- Casualty model: ~78–84%
- Blackout model: ~85–90%
- Triage model: ~87–92%

The variance in accuracy is because the noise terms in the score functions create inherently ambiguous cases near the decision boundaries. A model that claims 100% accuracy on this data would be overfitting.

---

## 6. Application Layer

**File**: `app.py`

### 6.1 Page Configuration & CSS Theme

```python
st.set_page_config(layout="wide", ...)
```

`layout="wide"` uses the full browser width. This is essential for displaying the map and intelligence panel side-by-side without cramping.

**Theme**: Dark seismic aesthetic using CSS injected via `st.markdown()`.
- Background `#080d18` — very dark navy, easier on the eyes for emergency operations rooms
- Font `Barlow Condensed` — a condensed sans-serif allowing more data per line; used in military/emergency contexts in real systems
- Font `Share Tech Mono` — monospace for all numbers and alerts; mimics terminal/radar displays
- Color system: Red `#ff3b4e` = danger, Orange `#ff8c00` = warning, Green `#1dd176` = safe. These are the same semantics as traffic lights and international emergency signage.

---

### 6.2 Model Loading (Caching)

```python
@st.cache_resource
def load_models():
    return joblib.load("damage_model.pkl"), ...
```

`@st.cache_resource` tells Streamlit: "Load this function's result once and reuse it for every subsequent user interaction." Without caching, the models would be loaded from disk on EVERY slider interaction, causing 1–2 second delays. With caching, loading happens once at startup (~0.3 seconds) and all subsequent predictions are near-instantaneous.

---

### 6.3 Session State

```python
st.session_state.lat  = ...
st.session_state.lon  = ...
st.session_state.sim_step = ...
```

Streamlit re-runs the entire `app.py` script from top to bottom on every user interaction (slider move, button click, etc.). Without `session_state`, all variables would reset on every interaction — the map would jump to a new location, patients would be regenerated, etc.

`session_state` persists values across reruns. Key variables stored:
- `lat, lon` — current earthquake epicenter (doesn't jump when you move a slider)
- `sim_step` — current simulation step (0–8)
- `sim_running` — whether simulation is advancing
- `patients_df, buildings_df` — generated once per earthquake, not per slider move
- `eq_seed` — random seed for patient/building generation (same seed = same people every rerun)
- `response_log` — list of simulation event messages
- `hist_df` — cached history table

---

### 6.4 Patient Generation (Rich Profiles)

`gen_patients(lat, lon, n=22, seed=0)` generates patients using `np.random.RandomState(seed)`.

**Why use a seeded random state?**
`RandomState(seed)` creates a deterministic random number generator. Every call with the same seed and same lat/lon produces exactly the same 22 patients. This prevents the map from "flickering" — patients don't jump around as you move sliders. They only change when you click "New EQ" (which changes the seed).

**How patients get their triage labels:**
The 4 clinical features (`age, injury_severity, conscious, heart_rate`) are fed into the `triage_model` at runtime. The model returns 0/1/2 (integer), which is then decoded by `encoders["triage"].inverse_transform()` back to "Critical"/"Minor"/"Serious".

**Patient profiles include:**
- Full name (drawn from arrays of Indian first/last names, realistic for a project set in India at lat≈20°N, lon≈78°E)
- Occupation and pre-existing conditions (affects real-world treatment priority but displayed as context, not fed into ML model — a deliberate simplification)
- Vital signs including SpO₂ (blood oxygen saturation) — values 84–100%, where <95% indicates respiratory distress
- Blood pressure (systolic/diastolic) — e.g., "140/90"
- Rescue status — their current operational state

---

### 6.5 Building Damage Simulation

`gen_buildings()` uses a rule-based formula (not a separate ML model).

**Why rule-based, not ML, for buildings?**
Adding a 5th ML model would require another training dataset and encoder. Since building damage can be derived deterministically from magnitude, depth, and building properties using well-understood structural engineering principles, a formula is more transparent and easier to explain.

**Formula:**
```python
score = magnitude * 1.1      # shaking intensity
      - strength * 0.85      # structural resistance
      + floors * 0.09        # taller = more vulnerable to resonance
      + age_years * 0.025    # older buildings have material degradation
      - depth * 0.025        # deeper = less surface shaking
      + noise(σ=0.4)

if score > 8.2  → "Collapse"
if score > 5.8  → "Structural damage"
else            → "Minor damage"
```

**Reconstruction time estimates** are based on typical post-disaster reconstruction timelines:
- Minor (cosmetic + non-structural): 1–3 months
- Structural damage (load-bearing walls, foundation): 3–12 months
- Collapse (complete rebuild required): 1–3 years

---

### 6.6 Resource Allocation Formula

This is the most important section to understand for demos.

**Every resource number has an explicit formula based on ML outputs + triage counts.**

```
ambulances   = 2 + (n_critical × 2) + (n_serious × 1) + [1 if High casualty risk]

rescue_teams = 2 + [3 if High damage | 1 if Medium | 0 if Low] + n_collapsed_buildings

med_kits     = (n_critical × 5) + (n_serious × 3) + (n_minor × 1) + [10 buffer if High casualty]

helicopters  = [1 if any critical patients]
             + [1 if High damage (aerial survey)]
             + [1 if comms Failure (satellite relay)]

search_units = 2 + (n_collapsed × 2)

comms_trucks = 0 if comms Safe
             = 1 if comms Risk
             = 3 if comms Failure

field_hospitals = 1 if (n_critical + n_serious) > 8 else 0
```

**Why these specific numbers?**

- **Ambulances**: Each Critical patient needs a dedicated vehicle (they may need CPR/intervention en route). Serious patients can sometimes share. The +1 for High casualty risk pre-positions a spare unit anticipating new casualties discovered as rubble is cleared.

- **Rescue teams**: FEMA Urban Search and Rescue (US&R) protocols define Type III teams for low damage, Type II for medium, and Type I for high. The base of 2 ensures at minimum one search team and one support team. Each collapsed building gets its own dedicated team because collapse rescue is specialized and simultaneous — you can't have one team working two collapses at once.

- **Medical kits**: An Advanced Trauma Kit contains tourniquets, chest seals, hemostatic gauze, and airway adjuncts — needed for Critical. An Intermediate Kit has splints, IV access, wound care. A Basic Kit has bandages and analgesics. The 10-unit buffer under High casualty anticipates patients being found as rescue progresses.

- **Helicopters**: Minimum 1 for any critical patient (they may need airlift to a hospital with surgical capability). The second unit for High damage enables aerial reconnaissance to identify trapped survivors and direct ground teams. The third unit for comms Failure enables satellite communication relay (helicopters carry relay equipment in real operations).

- **Field hospitals**: Deploying a field hospital (essentially an inflatable surgical suite) is only warranted when local hospital capacity will be exceeded. The threshold of 8 severe patients (Critical + Serious) is based on typical district hospital surge capacity in India of ~6–10 emergency beds.

---

### 6.7 Explainability Engine

```python
def explain(magnitude, depth, pop_density, infrastructure, urban_val, time_val):
```

This function takes the raw input values and generates plain-English explanations of why the damage model made its prediction. It does NOT use SHAP (SHapley Additive exPlanations) or LIME — those libraries require the model internals and add complexity. Instead, it uses the known feature-importance relationships from the score function:

- If magnitude >= 7.0 → strong evidence for high damage
- If infrastructure <= 3 → "very weak infrastructure" label
- If depth < 20 → "shallow focus — maximum surface energy"
- etc.

Each reason maps directly to a feature value that falls outside the "neutral" range. Counter-reasons are generated for features that actually reduce damage risk.

This approach is called **domain-driven explainability** — using human knowledge of what features matter rather than post-hoc model interpretation. It's more intuitive for non-technical audiences.

---

### 6.8 Historical Simulation

`gen_history(n=10, seed=99)` generates 10 simulated past earthquake events.

For each event:
1. Random feature values are generated (magnitude, depth, etc.)
2. These are fed through the trained ML models
3. Predictions (damage, casualty, blackout) are returned and stored

This demonstrates that the models can classify ANY input, not just the current slider values. The trend chart shows magnitude (line) overlaid with damage level (bars), making the magnitude→damage correlation visually apparent.

---

### 6.9 Map System

**Library**: Folium (Python wrapper for Leaflet.js)
**Tile layer**: CartoDB dark_matter — a dark-themed tile set that makes colored markers pop visually

**Why static map key (`key=f"map_{seed}_{sim_step}"`):**
The map is rebuilt every time the simulation step advances OR a new earthquake is generated. The `key` parameter tells Streamlit's `st_folium` which version of the map to render. If the key doesn't change (e.g., user just moves a slider), the map is NOT rebuilt — this prevents the flickering/glitching seen in v2.

**What each element represents:**

| Element | Shape | Color | Meaning |
|---------|-------|-------|---------|
| Patient | Circle (radius 7px) | 🔴 Red | Critical triage |
| Patient | Circle (radius 7px) | 🟠 Orange | Serious triage |
| Patient | Circle (radius 7px) | 🟢 Green | Minor triage |
| Building | House icon | Red | Collapsed |
| Building | House icon | Orange | Structural damage |
| Building | House icon | Green | Minor/safe |
| Epicenter | Star icon | Black/Red | Earthquake source |
| Damage zone | 3 concentric circles | Varies | Impact radius |

**Clicking any marker** opens a popup with the full profile of that patient (name, vitals, triage level, rescue status) or building (type, floors, strength, reconstruction time).

**Shockwave rings**: Three concentric circles are drawn at 100%, 65%, and 35% of the impact radius, with decreasing opacity. This represents the realistic attenuation of seismic energy — the inner zone has the most intense shaking, outer zones are progressively safer.

---

### 6.10 Simulation Mode

The simulation represents the spread of a disaster response over time, not the spread of the earthquake itself (earthquakes are instantaneous — they don't "spread"). What spreads is:

1. **Knowledge**: As rescue teams move outward, they discover more victims
2. **Impact radius**: The "awareness circle" on the map grows as responders fan out
3. **Patient count**: More casualties are found as rubble is cleared

**Simulation step behavior:**
- Step 1: Event confirmed (seismic sensors)
- Step 2: Evacuation broadcast
- Step 3: Ambulances dispatched to known critical patients
- Step 4: Rescue teams deployed to damage zone
- Step 5: Field triage station established
- Step 6: Helicopter aerial survey + airlift
- Step 7: Backup comms activated
- Step 8: Full incident command operational

Each step adds 2 new patients (representing newly discovered casualties), and the map's impact radius grows by 15km (representing expanding search area).

**Why `time.sleep(0.7)` + `st.rerun()`:**
Streamlit doesn't have a native animation loop. The pattern used here: set `sim_running=True`, each script execution advances one step, then triggers the next execution after 0.7 seconds. This creates a 0.7s per step animation rate.

---

### 6.11 KPI Strip

Six metric cards displayed across the full width, each sourced differently:

| Card | Source |
|------|--------|
| Damage Severity | `damage_model.predict()` → decoded by encoder |
| Casualty Risk | `casualty_model.predict()` → decoded |
| Comms Blackout | `blackout_model.predict()` → decoded |
| Triage Summary | Count of `patients["triage"]` values |
| Buildings | Count of `buildings["damage"]` values |
| Resources | Sum of ambulances + rescue + helicopters |

The confidence percentage on ML cards comes from `predict_proba()` — the maximum probability across all classes. For example, `[0.78, 0.03, 0.19]` for ["High","Low","Medium"] gives confidence = 78%.

---

### 6.12 Alert System

Eight distinct alert conditions are checked in sequence:
1. `damage_pred == "High"` → red alert
2. `casualty_pred == "High"` → red alert
3. `blackout_pred == "Failure"` → red alert
4. `magnitude >= 7.5` → red alert (major event threshold)
5. `n_collapsed >= 3` → orange alert (buried survivor probability high)
6. `blackout_pred == "Risk"` → orange alert
7. `time_val == 1` (night) → orange alert
8. `n_critical > 5` → orange alert (surge threshold)

If none apply, a green "stable" message is shown. The color mapping (red/orange/green CSS classes) applies the emergency color system consistently throughout.

---

## 7. Every Dashboard Panel Explained

### Panel: ML Classification Results (KPI Strip)
**What**: 6 colored metric cards at the top
**Data source**: Live ML model predictions from current sidebar inputs
**How to read**: Red = immediate action needed. Yellow/Orange = monitor closely. Green = manageable.
**Confidence %**: Higher % means the model is more certain. <60% = borderline case.

### Panel: Disaster Map
**What**: Interactive dark map with epicenter, damage rings, patient dots, building icons
**Data source**: epicenter from session state, patients from triage model, buildings from rule formula
**How to interact**: Click any dot or house icon to see full details of that patient/building

### Panel: Active Alerts
**What**: Real-time conditional alert messages
**Data source**: Derived from ML predictions + computed triage counts
**How to read**: Priority order top-to-bottom. Red = act now. Orange = prepare.

### Panel: Why This Prediction?
**What**: Plain-English explanation of the Damage model's classification
**Data source**: Rule-based interpretation of feature values
**Purpose**: Allows non-technical stakeholders to audit and trust the prediction

### Panel: Model Confidence (Bar Charts)
**What**: Three bar charts showing probability distribution across all classes
**Data source**: `predict_proba()` for each ML model
**How to read**: The highlighted bar is the predicted class. A bar at 90%+ = high confidence. Multiple bars near 33% = very uncertain prediction.

### Panel: Resource Allocation
**What**: 7 resource type cards with counts and calculation formula shown underneath
**Data source**: Formula using ML predictions + triage counts (see Section 6.6)
**How to demo**: Change magnitude from 5.0 to 8.0 and watch all resource numbers increase in real-time

### Panel: Simulation Response Log
**What**: Timeline of response actions taken (only appears after clicking ▶ Simulate)
**Data source**: Hardcoded message strings triggered at each simulation step
**Purpose**: Shows how an ICS (Incident Command System) would respond over time

### Panel: Patient Triage Queue
**What**: Sortable table of all patients + triage classification chart
**Data source**: Triage ML model applied to each patient's vitals
**How to read**: Patients sorted Critical first. Click map markers for richer profiles.

### Panel: Building Damage Assessment
**What**: Donut chart + table of all buildings
**Data source**: Rule-based formula using magnitude, depth, strength, floors, age
**Reconstruction column**: Time estimate for repair/rebuilding

### Panel: Historical Simulation (Last 10 Events)
**What**: Line+bar chart of past 10 simulated earthquakes
**Data source**: 10 randomly generated scenarios fed through all ML models
**Purpose**: Shows model behavior across different input combinations; demonstrates the magnitude→damage trend

---

## 8. Demo Guide

### How to run a compelling live demo

**Step 1 — Show baseline (safe scenario)**
Set sliders: Magnitude=5.0, Depth=80, Infra=8, Pop=500, Rural, Day
Observe: Low damage, Low casualties, Safe comms, green cards everywhere

**Step 2 — Escalate to disaster**
Gradually increase: Magnitude=7.8, Depth=15, Infra=2, Pop=8000, Urban, Night
Observe: All cards flip to Red, resource numbers jump dramatically, alerts appear
This shows the model responds sensibly to input changes

**Step 3 — Click a patient on the map**
Point out: "Each dot is a specific person — name, age, vitals, injury type, blood group"
This demonstrates the richness of the triage simulation

**Step 4 — Point to resource allocation**
Say: "These aren't random numbers. Ambulances = 2 + (critical×2) + (serious×1)..."
Show the formula text under each resource card

**Step 5 — Run simulation**
Click ▶ Simulate in the sidebar
Watch the map radius expand step by step
Watch the response log populate with timestamped actions
Watch patient count grow (new discoveries as rubble is cleared)

**Step 6 — Show confidence bars**
Point to the model confidence charts
"When magnitude is borderline (M6.5), the Medium and High bars are close — the model is uncertain. That's honest ML behavior."

**Step 7 — Show historical trends**
Click "Regenerate History" a few times
"Every time, the models classify these 10 different events. Notice: higher magnitude events tend to cluster in the red/orange damage bars"

**Step 8 — Show explainability**
Change to a High damage prediction
"The model says High. Why? Shallow depth, weak infrastructure, urban area. We can audit every decision."

---

## 9. Design Decisions FAQ

**Q: Why are patients shown as colored circles and not icons?**
A: Folium's CircleMarker renders identically regardless of zoom level and supports custom colors easily. Icon markers (like the building houses) have a fixed Leaflet icon library that doesn't support arbitrary hex colors — so circles were chosen for patients to get the exact triage colors.

**Q: Why does the map use CartoDB dark_matter tiles instead of Google Maps?**
A: CartoDB dark_matter is available without an API key (it's a free public tile server). The dark base makes colored markers dramatically more visible than on a bright terrain map. It also fits the emergency operations room aesthetic.

**Q: Why not use SHAP for explainability?**
A: SHAP requires the full model internals and adds a dependency that can be slow to compute for GradientBoosting models. For a demo application, domain-driven rule explanations are faster and more interpretable for non-ML audiences.

**Q: Why are buildings rule-based instead of a 5th ML model?**
A: Building damage can be derived from structural engineering principles that are well understood and don't need learned from data. A 5th model would add complexity (another training set, another encoder, another pkl file) with no improvement in realism.

**Q: Why does the simulation add patients each step?**
A: This represents a real phenomenon: immediately after a major earthquake, the initial casualty count is always underestimated. As rescue teams penetrate rubble and search expanding areas, new survivors and casualties are continuously discovered. The first hours of a major event typically see the casualty count increase 3–5x.

**Q: Why Indian names and coordinates?**
A: The default coordinates (lat≈20°N, lon≈78°E) correspond to central India — a seismically active region (ISZ Zone III-IV by BIS 1893). The patient names are Indian to match this geographic context.

**Q: What is predict_proba() and why does confidence sometimes seem low?**
A: `predict_proba()` returns the model's estimated probability for each class. When inputs fall near a decision boundary (e.g., magnitude=6.5 on a High/Medium boundary), both classes get significant probability, and neither may exceed 65%. This is correct behavior — the model is expressing genuine uncertainty. A model that always outputs 95%+ confidence would be overconfident/overfit.

---

## 10. File Structure

```
quakeguard/
├── app.py                  # Main Streamlit dashboard (all UI + logic)
├── models.py               # Model training script (run to retrain)
├── data_simulation.py      # Synthetic data generation
├── requirements.txt        # Python dependencies with versions
├── README.md               # This document
├── LICENSE                 # MIT
│
├── earthquake_data.csv     # 1500-row earthquake training dataset
├── patient_data.csv        # 300-row patient training dataset
│
├── damage_model.pkl        # Trained GradientBoostingClassifier (damage)
├── casualty_model.pkl      # Trained RandomForestClassifier (casualty)
├── blackout_model.pkl      # Trained GradientBoostingClassifier (blackout)
├── triage_model.pkl        # Trained RandomForestClassifier (triage)
└── encoders.pkl            # Dict of 4 LabelEncoder objects
```

---

## 11. Limitations & Future Work

### Current Limitations

1. **Synthetic data**: All training data is simulated. Real-world earthquake damage depends on soil type (liquefaction), building codes, aftershock sequences, and emergency response capacity — none of which are modeled here.

2. **Static triage**: Patient vitals are generated once. In reality, vitals change over time (a Minor patient can deteriorate to Critical within minutes from internal bleeding).

3. **No geospatial awareness**: The model treats all earthquakes at the same geographic point. Real damage depends on fault type (strike-slip vs thrust), directivity, and basin amplification.

4. **Single-event scope**: The simulation doesn't model aftershocks, which cause additional damage and complicate rescue operations (rescuers must stop work during aftershocks).

5. **Resource deployment**: Resources are allocated but not routed. A full system would need shortest-path routing from resource depots to patient locations.

### Potential Extensions

- Connect to USGS Earthquake Feed API for real events
- Add SHAP explainability for individual patient triage decisions
- Time-series triage: vitals deteriorate stochastically during simulation
- Multi-event mode: simulate aftershock sequences
- Hospital routing: find nearest hospital with capacity for each critical patient
- Export to PDF: generate a PDF incident report from current simulation state

---

## License

MIT — see [LICENSE](LICENSE).

# ==========================================
# QuakeGuard v3 - Earthquake Response Dashboard
# ==========================================
# Run with:  streamlit run app.py
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import warnings
from datetime import datetime, timedelta
import random

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="QuakeGuard v3",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────
# THEME / CSS
# ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
  font-family: 'Barlow Condensed', sans-serif;
  background-color: #080d18;
  color: #d4e4f4;
}
.stApp { background: #080d18; }

/* ── cards ── */
.kpi-card {
  background: linear-gradient(160deg,#0d1626,#101e30);
  border: 1px solid #172d4a;
  border-radius: 10px;
  padding: 16px 12px;
  text-align: center;
  margin-bottom: 8px;
}
.kpi-label {
  font-size: 10px;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  color: #4a7a9b;
  margin-bottom: 6px;
}
.kpi-value {
  font-family: 'Share Tech Mono', monospace;
  font-size: 22px;
  font-weight: 700;
  line-height: 1.1;
}
.kpi-sub {
  font-size: 11px;
  color: #4a7a9b;
  margin-top: 4px;
}

/* ── severity colours ── */
.c-red    { color: #ff3b4e; }
.c-orange { color: #ff8c00; }
.c-green  { color: #1dd176; }
.c-blue   { color: #1e90ff; }
.c-white  { color: #d4e4f4; }

/* ── alert rows ── */
.alert-red  { background:rgba(255,59,78,.10); border-left:3px solid #ff3b4e;
              border-radius:5px; padding:8px 14px; margin:4px 0;
              font-family:'Share Tech Mono',monospace; font-size:12px; }
.alert-orange{ background:rgba(255,140,0,.10); border-left:3px solid #ff8c00;
               border-radius:5px; padding:8px 14px; margin:4px 0;
               font-family:'Share Tech Mono',monospace; font-size:12px; }
.alert-green { background:rgba(29,209,118,.08); border-left:3px solid #1dd176;
               border-radius:5px; padding:8px 14px; margin:4px 0;
               font-family:'Share Tech Mono',monospace; font-size:12px; }

/* ── section header ── */
.sec-hdr {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: #4a7a9b;
  border-bottom: 1px solid #172d4a;
  padding-bottom: 5px;
  margin: 18px 0 10px 0;
}

/* ── patient card ── */
.pt-card {
  background: #0d1626;
  border: 1px solid #172d4a;
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 6px;
  font-size: 13px;
}
.pt-name { font-weight: 700; font-size: 15px; }
.pt-badge {
  display: inline-block;
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
}
.badge-critical { background:#ff3b4e22; color:#ff3b4e; border:1px solid #ff3b4e55; }
.badge-serious  { background:#ff8c0022; color:#ff8c00; border:1px solid #ff8c0055; }
.badge-minor    { background:#1dd17622; color:#1dd176; border:1px solid #1dd17655; }

/* ── resource card ── */
.res-card {
  background: #0d1626;
  border: 1px solid #172d4a;
  border-radius: 8px;
  padding: 12px 8px;
  text-align: center;
}
.res-num  { font-family:'Share Tech Mono',monospace; font-size:28px; font-weight:700; }
.res-lbl  { font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color:#4a7a9b; }
.res-why  { font-size: 11px; color: #3a6080; margin-top:4px; }

/* ── sim badge ── */
.sim-pill {
  display: inline-block;
  background: #ff3b4e;
  color: white;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 2px;
  padding: 3px 12px;
  border-radius: 20px;
  animation: blink 1.4s ease-in-out infinite;
  font-family: 'Share Tech Mono', monospace;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:.45;} }

/* ── sidebar ── */
[data-testid="stSidebar"] {
  background: #060b14 !important;
  border-right: 1px solid #172d4a;
}
.stSlider > div > div { background: #172d4a !important; }

/* ── buttons ── */
.stButton > button {
  background: linear-gradient(135deg,#0d1e35,#091628) !important;
  color: #d4e4f4 !important;
  border: 1px solid #1e3d5c !important;
  border-radius: 6px !important;
  font-family: 'Share Tech Mono', monospace !important;
  font-size: 12px !important;
  letter-spacing: 1px !important;
}
.stButton > button:hover {
  border-color: #1e90ff !important;
  box-shadow: 0 0 10px rgba(30,144,255,.25) !important;
}

/* ── tables ── */
.stDataFrame thead { background: #0d1626 !important; }

/* ── title ── */
.title-wrap { margin-bottom: 6px; }
.title-main {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 36px;
  font-weight: 800;
  letter-spacing: 2px;
  text-shadow: 0 0 40px rgba(30,144,255,.35);
}
.title-sub { font-size: 13px; color:#4a7a9b; letter-spacing: 3px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# LOAD MODELS  (cached)
# ──────────────────────────────────────────
@st.cache_resource
def load_models():
    return (
        joblib.load("damage_model.pkl"),
        joblib.load("casualty_model.pkl"),
        joblib.load("blackout_model.pkl"),
        joblib.load("triage_model.pkl"),
        joblib.load("encoders.pkl"),
    )

damage_model, casualty_model, blackout_model, triage_model, encoders = load_models()

EQ_FEATS  = ["magnitude","depth","population_density","infrastructure","urban","time_night","distance"]
PAT_FEATS = ["age","injury_severity","conscious","heart_rate"]

# ──────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────
FIRST_NAMES = ["Arjun","Priya","Ravi","Sunita","Vikram","Meena","Aditya","Kavya",
               "Rahul","Ananya","Sanjay","Deepa","Karthik","Lakshmi","Nikhil","Pooja",
               "Suresh","Divya","Manoj","Rekha","Amit","Shreya","Dinesh","Usha",
               "Rohit","Swathi","Mohan","Radha","Venkat","Geetha","Harish","Nandini"]
LAST_NAMES  = ["Sharma","Patel","Reddy","Nair","Iyer","Rao","Kumar","Singh",
               "Verma","Pillai","Mehta","Joshi","Gupta","Bhat","Chaudhary","Desai"]
INJURY_TYPES = {1:"Lacerations / bruises", 2:"Fracture / internal bleeding", 3:"Crush injury / trauma"}
BLOOD_GROUPS = ["A+","A-","B+","B-","O+","O-","AB+","AB-"]
CONDITIONS   = ["None","Diabetes","Hypertension","Asthma","Heart disease"]
OCCUPATIONS  = ["Student","Teacher","Farmer","Engineer","Doctor","Shopkeeper",
                "Construction worker","Homemaker","Govt employee","Driver"]
BLDG_TYPES   = ["Residential","Commercial","Hospital","School","Industrial","Government"]
RESCUE_STATUS= ["Awaiting rescue","Rescue in progress","Evacuated","Field treated"]

# ──────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────
def init():
    defs = dict(
        lat=20.0, lon=78.0,
        sim_step=0, sim_running=False,
        patients_df=None, buildings_df=None,
        hist_df=None, eq_seed=0,
        response_log=[],
    )
    for k,v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
init()

# ──────────────────────────────────────────
# DATA GENERATORS
# ──────────────────────────────────────────
def gen_patients(lat, lon, n=22, seed=0):
    rng = np.random.RandomState(seed)
    ages   = rng.randint(1, 88, n)
    sevs   = rng.choice([1,2,3], n, p=[0.38,0.35,0.27])
    consc  = rng.choice([0,1], n, p=[0.28,0.72])
    hr     = rng.randint(55, 148, n)
    bps    = rng.randint(80, 180, n)
    bpd    = rng.randint(50, 110, n)
    spo2   = rng.randint(84, 100, n)
    first  = rng.choice(FIRST_NAMES, n)
    last   = rng.choice(LAST_NAMES, n)

    df = pd.DataFrame({
        "name":          [f"{a} {b}" for a,b in zip(first,last)],
        "age":           ages,
        "gender":        rng.choice(["Male","Female"], n),
        "blood_group":   rng.choice(BLOOD_GROUPS, n),
        "occupation":    rng.choice(OCCUPATIONS, n),
        "pre_condition": rng.choice(CONDITIONS, n, p=[0.55,0.12,0.15,0.08,0.10]),
        "injury_severity": sevs,
        "injury_type":   [INJURY_TYPES[s] for s in sevs],
        "conscious":     consc,
        "heart_rate":    hr,
        "bp":            [f"{s}/{d}" for s,d in zip(bps,bpd)],
        "spo2":          spo2,
        "rescue_status": rng.choice(RESCUE_STATUS, n, p=[0.35,0.30,0.20,0.15]),
        "lat":           lat + rng.uniform(-0.30, 0.30, n),
        "lon":           lon + rng.uniform(-0.30, 0.30, n),
    })
    preds = triage_model.predict(df[PAT_FEATS])
    df["triage"] = encoders["triage"].inverse_transform(preds)
    return df


def gen_buildings(lat, lon, n=16, seed=0, magnitude=6.0, depth=30.0):
    rng = np.random.RandomState(seed + 100)
    strength = rng.randint(1, 11, n)
    floors   = rng.randint(1, 18, n)
    age_yr   = rng.randint(2, 55, n)
    btype    = rng.choice(BLDG_TYPES, n)

    def dmg(s, f, a):
        score = (
            magnitude * 1.1
            - s * 0.85
            + f * 0.09
            + a * 0.025
            - depth * 0.025
            + rng.normal(0, 0.4)
        )
        if score > 8.2: return "Collapse"
        elif score > 5.8: return "Structural"
        else: return "Minor"

    damages = [dmg(s,f,a) for s,f,a in zip(strength,floors,age_yr)]
    recon   = {"Minor":"1–3 months","Structural":"3–12 months","Collapse":"1–3 years"}

    return pd.DataFrame({
        "type":           btype,
        "floors":         floors,
        "strength":       strength,
        "age_years":      age_yr,
        "damage":         damages,
        "reconstruction": [recon[d] for d in damages],
        "lat":            lat + rng.uniform(-0.32, 0.32, n),
        "lon":            lon + rng.uniform(-0.32, 0.32, n),
    })


# ──────────────────────────────────────────
# RESOURCE ALLOCATION  (formula-based)
# ──────────────────────────────────────────
# All numbers derive from:
#   damage_score    = {"Low":0, "Medium":1, "High":2}
#   casualty_score  = {"Low":0, "Medium":1, "High":2}
#   blackout_score  = {"Safe":0, "Risk":1, "Failure":2}
#   n_critical, n_serious, n_minor from triage model
#   n_collapsed from building damage rule
#
# Formula references:
#   WHO Emergency Medical Teams guidelines (basic unit sizing)
#   FEMA Search-and-Rescue team activation levels
# ──────────────────────────────────────────
SCORE_MAP   = {"Low":0,"Medium":1,"High":2,"Safe":0,"Risk":1,"Failure":2,"Minor":0,"Structural":1,"Collapse":2}

def compute_resources(damage_pred, casualty_pred, blackout_pred,
                      n_critical, n_serious, n_minor, n_collapsed):
    ds = SCORE_MAP[damage_pred]
    cs = SCORE_MAP[casualty_pred]
    bs = SCORE_MAP[blackout_pred]

    # ── Ambulances ──────────────────────────────────────────────
    # Base: 2 for any event.
    # +2 per Critical patient (each needs dedicated transport)
    # +1 per Serious patient  (may share)
    # +1 if High casualty risk (pre-position extra)
    ambulances = 2 + (n_critical * 2) + (n_serious * 1) + (1 if cs == 2 else 0)

    # ── Rescue teams ────────────────────────────────────────────
    # Base: 2 teams always on standby.
    # +3 if High damage (major structural search needed)
    # +1 per collapsed building (each collapse = 1 dedicated team)
    # +1 if night (slower ops, more teams needed)
    rescue_teams = 2 + (3 if ds == 2 else 1 if ds == 1 else 0) + n_collapsed

    # ── Medical kits ────────────────────────────────────────────
    # 5 per Critical (advanced trauma kit)
    # 3 per Serious  (intermediate kit)
    # 1 per Minor    (basic first aid)
    # +10 buffer if High casualty
    med_kits = (n_critical * 5) + (n_serious * 3) + (n_minor * 1) + (10 if cs == 2 else 0)

    # ── Helicopters ─────────────────────────────────────────────
    # 1 if any Critical patients exist
    # +1 if High damage (aerial survey + heavy lift)
    # +1 if comms Failure (satellite relay)
    helicopters = (1 if n_critical > 0 else 0) + (1 if ds == 2 else 0) + (1 if bs == 2 else 0)

    # ── Search units ────────────────────────────────────────────
    # 2 base + 2 per collapsed building
    search_units = 2 + (n_collapsed * 2)

    # ── Comms trucks ────────────────────────────────────────────
    # Only deployed if comms Risk or Failure
    comms_trucks = 0 if bs == 0 else (1 if bs == 1 else 3)

    # ── Field hospitals ─────────────────────────────────────────
    # Deployed when casualties are high enough to overwhelm nearby facilities
    field_hospitals = 1 if (n_critical + n_serious) > 8 else 0

    return dict(
        ambulances=ambulances,
        rescue_teams=rescue_teams,
        med_kits=med_kits,
        helicopters=helicopters,
        search_units=search_units,
        comms_trucks=comms_trucks,
        field_hospitals=field_hospitals,
    )


def resource_formula_text(damage_pred, casualty_pred, blackout_pred,
                           n_critical, n_serious, n_minor, n_collapsed):
    ds = SCORE_MAP[damage_pred]
    cs = SCORE_MAP[casualty_pred]
    bs = SCORE_MAP[blackout_pred]
    lines = {
        "ambulances":  f"2 base + {n_critical}×2 (critical) + {n_serious}×1 (serious)" +
                       (" + 1 (high casualty flag)" if cs==2 else ""),
        "rescue_teams":f"2 base + {'3' if ds==2 else '1' if ds==1 else '0'} (damage level) + {n_collapsed} (collapsed buildings)",
        "med_kits":    f"{n_critical}×5 + {n_serious}×3 + {n_minor}×1" +
                       (" + 10 buffer" if cs==2 else ""),
        "helicopters": f"{'1 (critical exist)' if n_critical>0 else '0'} + {'1 (high damage)' if ds==2 else '0'} + {'1 (comms failure)' if bs==2 else '0'}",
        "search_units":f"2 base + {n_collapsed}×2 collapsed",
        "comms_trucks":f"{'0 (comms OK)' if bs==0 else '1 (comms risk)' if bs==1 else '3 (comms failure)'}",
        "field_hospitals": f"{'1 — >8 severe casualties' if (n_critical+n_serious)>8 else '0 — <8 severe casualties'}",
    }
    return lines


# ──────────────────────────────────────────
# EXPLAINABILITY
# ──────────────────────────────────────────
def explain(magnitude, depth, pop_density, infrastructure, urban_val, time_val):
    reasons, counter = [], []
    if magnitude >= 7.0:  reasons.append(f"High magnitude ({magnitude}) → major ground motion")
    elif magnitude >= 5.5: reasons.append(f"Moderate magnitude ({magnitude})")
    else: counter.append(f"Low magnitude ({magnitude})")

    if pop_density >= 5000: reasons.append(f"Dense population ({pop_density:,}/km²) → mass exposure")
    elif pop_density <= 500: counter.append(f"Sparse population ({pop_density:,}/km²)")

    if infrastructure <= 3: reasons.append(f"Very weak infrastructure (score {infrastructure}/10)")
    elif infrastructure <= 5: reasons.append(f"Below-average infrastructure ({infrastructure}/10)")
    elif infrastructure >= 8: counter.append(f"Strong infrastructure ({infrastructure}/10)")

    if depth < 20: reasons.append(f"Shallow depth ({depth:.0f} km) → maximum surface shaking")
    elif depth > 70: counter.append(f"Deep focus ({depth:.0f} km) → energy absorbed before surface")

    if urban_val: reasons.append("Urban area — dense buildings amplify casualties")
    else: counter.append("Rural area — lower building density")

    if time_val: reasons.append("Night-time — population sleeping, slower evacuation")
    return reasons, counter


# ──────────────────────────────────────────
# HISTORICAL GENERATOR
# ──────────────────────────────────────────
def gen_history(n=10, seed=99):
    rng = np.random.RandomState(seed)
    rows = []
    base = datetime.now() - timedelta(days=n*4)
    for i in range(n):
        mag   = round(rng.uniform(4.5, 7.8), 1)
        dep   = round(rng.uniform(10, 90), 1)
        pop   = int(rng.randint(500, 8000))
        infra = int(rng.randint(2, 9))
        urb   = int(rng.randint(0, 2))
        ngt   = int(rng.randint(0, 2))
        dist  = round(rng.uniform(10, 200), 1)
        inp   = np.array([[mag,dep,pop,infra,urb,ngt,dist]])
        d = encoders["damage"].inverse_transform(damage_model.predict(inp))[0]
        c = encoders["casualty_risk"].inverse_transform(casualty_model.predict(inp))[0]
        b = encoders["blackout"].inverse_transform(blackout_model.predict(inp))[0]
        rows.append({"Date":(base+timedelta(days=i*4)).strftime("%b %d"),
                     "Mag":mag,"Damage":d,"Casualty":c,"Comms":b,"Location":f"Zone {chr(65+i)}"})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────
# MAP BUILDER  (static — no rerun flicker)
# ──────────────────────────────────────────
def build_map(lat, lon, radius_m, damage_pred, patients, buildings, sim_step):
    m = folium.Map(location=[lat, lon], zoom_start=7,
                   tiles="CartoDB dark_matter", prefer_canvas=True)

    ring_color = {"High":"#ff3b4e","Medium":"#ff8c00","Low":"#1dd176"}.get(damage_pred,"#1e90ff")

    # Shockwave rings (3 concentric, fading opacity)
    for frac, op, wt in [(1.0, 0.30, 2), (0.65, 0.17, 1), (0.35, 0.09, 1)]:
        folium.Circle(
            location=[lat, lon],
            radius=radius_m * frac,
            color=ring_color,
            fill=True,
            fill_opacity=op,
            weight=wt,
            tooltip=f"Impact zone – {int(radius_m*frac/1000)} km radius",
        ).add_to(m)

    # Epicenter star
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(
            f"<b style='color:#ff3b4e'>🌋 EPICENTER</b><br>"
            f"Step {sim_step}/8 | Radius {radius_m/1000:.0f} km", max_width=200),
        icon=folium.Icon(color="red", icon="star", prefix="fa"),
    ).add_to(m)

    # ── Patients ──
    # Each marker shows the PERSON's name + triage + vitals in popup.
    # Shape = circle; colour = triage level (Red/Orange/Green).
    triage_color = {"Critical":"#ff3b4e","Serious":"#ff8c00","Minor":"#1dd176"}
    for _, r in patients.iterrows():
        col = triage_color.get(r["triage"], "#aaaaaa")
        popup_html = (
            f"<div style='font-family:monospace;font-size:12px;min-width:170px'>"
            f"<b>{r['name']}</b> &nbsp;"
            f"<span style='background:{col};color:#000;padding:1px 6px;border-radius:3px;"
            f"font-size:10px;font-weight:700'>{r['triage'].upper()}</span><br>"
            f"Age {r['age']} · {r['gender']} · {r['occupation']}<br>"
            f"🩸 {r['blood_group']} &nbsp; 🫀 HR {r['heart_rate']} &nbsp; SpO₂ {r['spo2']}%<br>"
            f"BP {r['bp']} &nbsp; Conscious: {'Yes' if r['conscious'] else '<b style=color:red>No</b>'}<br>"
            f"Injury: {r['injury_type']}<br>"
            f"Pre-existing: {r['pre_condition']}<br>"
            f"Status: <b>{r['rescue_status']}</b>"
            f"</div>"
        )
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=7,
            color=col,
            fill=True,
            fill_color=col,
            fill_opacity=0.88,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=230),
            tooltip=f"{r['name']} · {r['triage']}",
        ).add_to(m)

    # ── Buildings ──
    bldg_color = {"Collapse":"red","Structural":"orange","Minor":"green"}
    bldg_icon  = {"Collapse":"home","Structural":"home","Minor":"home"}
    for _, r in buildings.iterrows():
        col  = bldg_color.get(r["damage"],"blue")
        icon = bldg_icon.get(r["damage"],"home")
        popup_html = (
            f"<div style='font-family:monospace;font-size:12px'>"
            f"<b>{r['type']}</b><br>"
            f"Floors: {r['floors']} &nbsp; Age: {r['age_years']} yrs &nbsp; Strength: {r['strength']}/10<br>"
            f"Damage: <b style='color:{'#ff3b4e' if r['damage']=='Collapse' else '#ff8c00' if r['damage']=='Structural' else '#1dd176'}'>{r['damage']}</b><br>"
            f"Reconstruction: {r['reconstruction']}"
            f"</div>"
        )
        folium.Marker(
            [r["lat"], r["lon"]],
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{r['type']} · {r['damage']}",
            icon=folium.Icon(color=col, icon=icon, prefix="fa"),
        ).add_to(m)

    # Legend
    legend = """
    <div style='position:fixed;bottom:18px;left:18px;z-index:9999;
         background:#0d1626cc;border:1px solid #172d4a;border-radius:8px;
         padding:10px 14px;font-size:12px;color:#d4e4f4;
         font-family:Share Tech Mono,monospace;line-height:1.8'>
      <b>LEGEND</b><br>
      <span style='color:#ff3b4e'>●</span> Critical patient &nbsp;
      <span style='color:#ff8c00'>●</span> Serious &nbsp;
      <span style='color:#1dd176'>●</span> Minor<br>
      <span style='color:#ff3b4e'>⌂</span> Collapsed &nbsp;
      <span style='color:#ff8c00'>⌂</span> Structural &nbsp;
      <span style='color:#1dd176'>⌂</span> Safe bldg<br>
      ★ Epicenter (click any marker for details)
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m


# ──────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sec-hdr">⚙ EARTHQUAKE PARAMETERS</div>', unsafe_allow_html=True)
    magnitude          = st.slider("Magnitude (Richter)", 4.0, 8.5, 6.5, 0.1)
    depth              = st.slider("Depth (km)", 5.0, 100.0, 30.0, 1.0)
    population_density = st.slider("Population Density (/km²)", 50, 10000, 3000, 50)
    infrastructure     = st.slider("Infrastructure Quality (1–10)", 1, 10, 5)
    urban              = st.selectbox("Location Type", ["Urban","Rural"])
    time_night         = st.selectbox("Time of Day", ["Day","Night"])
    distance           = st.slider("Distance from Epicenter (km)", 1.0, 300.0, 50.0, 1.0)

    urban_val = 1 if urban=="Urban" else 0
    time_val  = 1 if time_night=="Night" else 0

    st.markdown('<div class="sec-hdr">🌋 SIMULATION CONTROLS</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    new_eq  = c1.button("🌋 New EQ", use_container_width=True)
    run_sim = c2.button("▶ Simulate", use_container_width=True)

    if new_eq:
        seed = random.randint(0, 9999)
        st.session_state.eq_seed     = seed
        st.session_state.lat         = round(20 + np.random.uniform(-9,9), 4)
        st.session_state.lon         = round(78 + np.random.uniform(-9,9), 4)
        st.session_state.sim_step    = 0
        st.session_state.sim_running = False
        st.session_state.patients_df = None
        st.session_state.buildings_df= None
        st.session_state.response_log= []

    if run_sim:
        st.session_state.sim_running = True
        st.session_state.sim_step    = 0
        st.session_state.response_log= []

    if st.session_state.sim_running and st.button("⏹ Stop", use_container_width=True):
        st.session_state.sim_running = False

    if st.session_state.sim_step > 0:
        pct = int(st.session_state.sim_step / 8 * 100)
        st.progress(pct, text=f"Simulation {pct}% — Step {st.session_state.sim_step}/8")

    st.markdown('<div class="sec-hdr">📅 HISTORY</div>', unsafe_allow_html=True)
    if st.button("🔄 Regenerate History", use_container_width=True):
        st.session_state.hist_df = gen_history(10, seed=random.randint(0,999))


# ──────────────────────────────────────────
# ML PREDICTIONS
# ──────────────────────────────────────────
inp = np.array([[magnitude, depth, population_density,
                 infrastructure, urban_val, time_val, distance]])

damage_pred   = encoders["damage"].inverse_transform(damage_model.predict(inp))[0]
casualty_pred = encoders["casualty_risk"].inverse_transform(casualty_model.predict(inp))[0]
blackout_pred = encoders["blackout"].inverse_transform(blackout_model.predict(inp))[0]

damage_proba   = damage_model.predict_proba(inp)[0]
casualty_proba = casualty_model.predict_proba(inp)[0]
blackout_proba = blackout_model.predict_proba(inp)[0]

# ──────────────────────────────────────────
# SIMULATION STATE
# ──────────────────────────────────────────
if st.session_state.sim_running and st.session_state.sim_step < 8:
    st.session_state.sim_step += 1
    # Log response action
    step = st.session_state.sim_step
    log_msgs = {
        1: "⚡ Seismic sensors triggered. Event confirmed at M" + str(magnitude),
        2: "📡 Emergency broadcast issued. Evacuation order for inner zone.",
        3: "🚑 Ambulances dispatched to critical patient locations.",
        4: "🧑‍🚒 Rescue teams deployed to " + str((damage_model.predict(inp)[0])) + " damage zone.",
        5: "🏥 Field triage station established. Patients being sorted.",
        6: "🚁 Helicopter conducting aerial survey + casualty airlift.",
        7: "🔌 Backup comms satellite activated. Coordination restored.",
        8: "✅ All resources deployed. Incident command post operational.",
    }
    st.session_state.response_log.append(f"[T+{step*10}min] {log_msgs.get(step,'')}")
elif st.session_state.sim_running and st.session_state.sim_step >= 8:
    st.session_state.sim_running = False

sim_step = st.session_state.sim_step
radius_m = (50 + sim_step * 15) * 1000  # base 50km, grows 15km/step

# ──────────────────────────────────────────
# PATIENTS & BUILDINGS (stable seed)
# ──────────────────────────────────────────
seed = st.session_state.eq_seed
if st.session_state.patients_df is None:
    st.session_state.patients_df  = gen_patients(st.session_state.lat, st.session_state.lon, seed=seed)
    st.session_state.buildings_df = gen_buildings(st.session_state.lat, st.session_state.lon,
                                                  seed=seed, magnitude=magnitude, depth=depth)

patients  = st.session_state.patients_df
buildings = st.session_state.buildings_df

# Live sim adds extra critical patients each step
if sim_step > 0:
    extra = sim_step * 2
    if len(patients) < 22 + extra:
        more = gen_patients(st.session_state.lat, st.session_state.lon,
                            n=extra, seed=seed + sim_step * 7)
        patients = pd.concat([patients, more], ignore_index=True)

n_critical = (patients["triage"] == "Critical").sum()
n_serious  = (patients["triage"] == "Serious").sum()
n_minor    = (patients["triage"] == "Minor").sum()
n_collapsed= (buildings["damage"] == "Collapse").sum()
n_struct   = (buildings["damage"] == "Structural").sum()
n_safe     = (buildings["damage"] == "Minor").sum()

resources  = compute_resources(damage_pred, casualty_pred, blackout_pred,
                                n_critical, n_serious, n_minor, n_collapsed)
res_formula= resource_formula_text(damage_pred, casualty_pred, blackout_pred,
                                    n_critical, n_serious, n_minor, n_collapsed)

# ──────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────
lat = st.session_state.lat
lon = st.session_state.lon

st.markdown(f"""
<div class="title-wrap">
  <span class="title-main">🌍 QUAKEGUARD</span>
  <span class="title-sub"> &nbsp; EARTHQUAKE RESPONSE SYSTEM v3</span><br>
  <span style='font-size:12px;color:#2a5070;font-family:Share Tech Mono,monospace'>
    📍 {lat:.3f}°N, {lon:.3f}°E &nbsp;·&nbsp; M{magnitude} &nbsp;·&nbsp; {depth:.0f}km depth &nbsp;·&nbsp;
    {datetime.now().strftime("%H:%M:%S")}
  </span>
</div>
""", unsafe_allow_html=True)

if st.session_state.sim_running or sim_step > 0:
    st.markdown(
        f'<span class="sim-pill">● LIVE SIM — STEP {sim_step}/8 — RADIUS {int(radius_m/1000)} KM</span>',
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────
# KPI STRIP
# ──────────────────────────────────────────
st.markdown('<div class="sec-hdr">📊 ML CLASSIFICATION RESULTS</div>', unsafe_allow_html=True)

def severity_color(label):
    return {"High":"c-red","Failure":"c-red","Collapse":"c-red",
            "Medium":"c-orange","Risk":"c-orange","Structural":"c-orange",
            "Low":"c-green","Safe":"c-green","Minor":"c-green"}.get(label,"c-white")

def emoji_level(label):
    return {"High":"🔴","Medium":"🟡","Low":"🟢",
            "Failure":"🔴","Risk":"🟡","Safe":"🟢",
            "Critical":"🔴","Serious":"🟡","Minor":"🟢",
            "Collapse":"🔴","Structural":"🟡"}.get(label,"⚪") + " " + label

k1,k2,k3,k4,k5,k6 = st.columns(6)

for col, label_val, label_text, conf_val, conf_classes in [
    (k1, damage_pred,   "🏗 Damage Severity",  damage_proba,   encoders["damage"].classes_),
    (k2, casualty_pred, "🚑 Casualty Risk",    casualty_proba, encoders["casualty_risk"].classes_),
    (k3, blackout_pred, "📡 Comms Blackout",   blackout_proba, encoders["blackout"].classes_),
]:
    conf_pct = int(max(conf_val)*100)
    with col:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">{label_text}</div>
          <div class="kpi-value {severity_color(label_val)}">{emoji_level(label_val)}</div>
          <div class="kpi-sub">Confidence: {conf_pct}%</div>
        </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">🏥 Triage Summary</div>
      <div class="kpi-value c-red">{n_critical} Critical</div>
      <div class="kpi-sub c-orange">{n_serious} Serious · {n_minor} Minor</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">🏠 Buildings</div>
      <div class="kpi-value c-red">{n_collapsed} Collapsed</div>
      <div class="kpi-sub">{n_struct} Damaged · {n_safe} Safe</div>
    </div>""", unsafe_allow_html=True)

with k6:
    total_res = resources["ambulances"] + resources["rescue_teams"] + resources["helicopters"]
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">📦 Resources Deployed</div>
      <div class="kpi-value c-blue">{total_res}</div>
      <div class="kpi-sub">units total</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────
# MAIN LAYOUT: MAP (left) + INTEL (right)
# ──────────────────────────────────────────
map_col, intel_col = st.columns([3, 2], gap="medium")

with map_col:
    st.markdown('<div class="sec-hdr">🗺 DISASTER MAP  <span style="font-size:11px;color:#2a5070">(click any marker for full details)</span></div>', unsafe_allow_html=True)

    # Build map ONCE per state change; no auto-rerun from map interaction
    m = build_map(lat, lon, radius_m, damage_pred, patients, buildings, sim_step)
    st_folium(m, width=None, height=520, returned_objects=[], key=f"map_{seed}_{sim_step}")

with intel_col:

    # ── ALERTS ──────────────────────────────────
    st.markdown('<div class="sec-hdr">🚨 ACTIVE ALERTS</div>', unsafe_allow_html=True)
    alerts = []
    if damage_pred == "High":    alerts.append(("red",  "🔴 HIGH DAMAGE ZONE — Deploy rescue teams immediately"))
    if casualty_pred == "High":  alerts.append(("red",  "🔴 HIGH CASUALTY RISK — Mass casualty protocol active"))
    if blackout_pred == "Failure":alerts.append(("red", "🔴 COMMS BLACKOUT — Satellite relay required"))
    if magnitude >= 7.5:         alerts.append(("red",  f"🔴 MAJOR EVENT M{magnitude} — National emergency level"))
    if n_collapsed >= 3:         alerts.append(("orange",f"🟡 {n_collapsed} COLLAPSED BUILDINGS — Buried survivors likely"))
    if blackout_pred == "Risk":  alerts.append(("orange","🟡 COMMS DEGRADED — Backup channels advised"))
    if time_val == 1:            alerts.append(("orange","🟡 NIGHT OPERATION — Visibility impaired"))
    if n_critical > 5:           alerts.append(("orange",f"🟡 {n_critical} CRITICAL PATIENTS — Surge capacity needed"))
    if not alerts:               alerts.append(("green", "🟢 SITUATION STABLE — Continue monitoring"))

    for (level, msg) in alerts:
        st.markdown(f'<div class="alert-{level}">{msg}</div>', unsafe_allow_html=True)

    # ── EXPLAINABILITY ───────────────────────────
    st.markdown('<div class="sec-hdr">🔍 WHY THIS PREDICTION?</div>', unsafe_allow_html=True)
    reasons, counter = explain(magnitude, depth, population_density, infrastructure, urban_val, time_val)
    st.markdown(f"<div style='font-size:12px;color:#4a7a9b;margin-bottom:6px'>Damage → <b style='color:#d4e4f4'>{damage_pred}</b></div>", unsafe_allow_html=True)
    for r in reasons:
        st.markdown(f'<div class="alert-orange" style="border-color:#ff8c00">✔ {r}</div>', unsafe_allow_html=True)
    for c in counter:
        st.markdown(f'<div class="alert-green">✦ {c}</div>', unsafe_allow_html=True)

    # ── CONFIDENCE BARS ──────────────────────────
    st.markdown('<div class="sec-hdr">📈 MODEL CONFIDENCE</div>', unsafe_allow_html=True)
    for name, pred, proba, classes in [
        ("Damage",   damage_pred,   damage_proba,   encoders["damage"].classes_),
        ("Casualty", casualty_pred, casualty_proba, encoders["casualty_risk"].classes_),
        ("Blackout", blackout_pred, blackout_proba, encoders["blackout"].classes_),
    ]:
        bar_colors = ["#ff3b4e" if c == pred else "#172d4a" for c in classes]
        fig = go.Figure(go.Bar(
            x=list(classes), y=[round(p*100,1) for p in proba],
            marker_color=bar_colors,
            text=[f"{p*100:.0f}%" for p in proba],
            textposition="outside",
            textfont=dict(color="#d4e4f4", size=10),
        ))
        fig.update_layout(
            height=110, margin=dict(l=0,r=0,t=18,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#4a7a9b", tickfont=dict(size=10)),
            yaxis=dict(showticklabels=False, showgrid=False, range=[0,115]),
            title=dict(text=name, font=dict(color="#4a7a9b",size=11)),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────
# RESOURCE ALLOCATION
# ──────────────────────────────────────────
st.markdown('<div class="sec-hdr">📦 RESOURCE ALLOCATION  <span style="font-size:11px;color:#2a5070">(formula-derived from ML outputs + triage counts)</span></div>', unsafe_allow_html=True)

res_items = [
    ("🚑", "Ambulances",     resources["ambulances"],     res_formula["ambulances"]),
    ("🧑‍🚒", "Rescue Teams",  resources["rescue_teams"],   res_formula["rescue_teams"]),
    ("🏥", "Med Kits",       resources["med_kits"],       res_formula["med_kits"]),
    ("🚁", "Helicopters",    resources["helicopters"],    res_formula["helicopters"]),
    ("🔦", "Search Units",   resources["search_units"],   res_formula["search_units"]),
    ("📡", "Comms Trucks",   resources["comms_trucks"],   res_formula["comms_trucks"]),
    ("🏗", "Field Hospitals",resources["field_hospitals"],res_formula["field_hospitals"]),
]

r_cols = st.columns(7)
for col, (emoji, label, val, formula) in zip(r_cols, res_items):
    with col:
        st.markdown(f"""<div class="res-card">
          <div style="font-size:22px">{emoji}</div>
          <div class="res-num c-blue">{val}</div>
          <div class="res-lbl">{label}</div>
          <div class="res-why">{formula}</div>
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# SIMULATION RESPONSE LOG
# ──────────────────────────────────────────
if st.session_state.response_log:
    st.markdown('<div class="sec-hdr">🔁 SIMULATION RESPONSE LOG</div>', unsafe_allow_html=True)
    for entry in st.session_state.response_log:
        level = "alert-red" if "🚑" in entry or "🚁" in entry else "alert-orange" if "🧑" in entry else "alert-green"
        st.markdown(f'<div class="{level}">{entry}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────
# PATIENT DETAIL TABLE
# ──────────────────────────────────────────
st.markdown('<div class="sec-hdr">🏥 PATIENT TRIAGE QUEUE  <span style="font-size:11px;color:#2a5070">(click map markers for full profiles)</span></div>', unsafe_allow_html=True)

pt_tab1, pt_tab2 = st.tabs(["📋 Priority List", "📊 Distribution"])

with pt_tab1:
    porder = {"Critical":0,"Serious":1,"Minor":2}
    pt_display = patients.copy()
    pt_display["_p"] = pt_display["triage"].map(porder)
    pt_display = pt_display.sort_values("_p").drop("_p",axis=1)
    pt_display = pt_display.drop(columns=["lat","lon"], errors="ignore")
    st.dataframe(pt_display, use_container_width=True, height=300)

with pt_tab2:
    tri_counts = patients["triage"].value_counts()
    fig_tri = go.Figure(go.Bar(
        x=tri_counts.index,
        y=tri_counts.values,
        marker_color=["#ff3b4e" if l=="Critical" else "#ff8c00" if l=="Serious" else "#1dd176"
                      for l in tri_counts.index],
        text=tri_counts.values, textposition="outside",
        textfont=dict(color="#d4e4f4"),
    ))
    fig_tri.update_layout(
        height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#4a7a9b"), yaxis=dict(color="#4a7a9b",showgrid=False),
        margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
    )
    st.plotly_chart(fig_tri, use_container_width=True)

# ──────────────────────────────────────────
# BUILDING DAMAGE
# ──────────────────────────────────────────
st.markdown('<div class="sec-hdr">🏗 BUILDING DAMAGE ASSESSMENT</div>', unsafe_allow_html=True)

bl1, bl2 = st.columns([1,2])
with bl1:
    dmg_c = buildings["damage"].value_counts()
    fig_b = go.Figure(go.Pie(
        labels=dmg_c.index, values=dmg_c.values, hole=0.52,
        marker_colors=["#ff3b4e" if l=="Collapse" else "#ff8c00" if l=="Structural" else "#1dd176"
                       for l in dmg_c.index],
        textfont=dict(size=11,color="#d4e4f4"),
    ))
    fig_b.update_layout(
        height=230, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#4a7a9b",size=10)),
    )
    st.plotly_chart(fig_b, use_container_width=True)

with bl2:
    bld_display = buildings.sort_values("damage",key=lambda x:x.map({"Collapse":0,"Structural":1,"Minor":2}))
    bld_display = bld_display.drop(columns=["lat","lon"],errors="ignore")
    st.dataframe(bld_display, use_container_width=True, height=230)

# ──────────────────────────────────────────
# HISTORICAL SIMULATION
# ──────────────────────────────────────────
st.markdown('<div class="sec-hdr">📅 HISTORICAL SIMULATION — LAST 10 EVENTS</div>', unsafe_allow_html=True)

if st.session_state.hist_df is None:
    st.session_state.hist_df = gen_history(10)

hist = st.session_state.hist_df

h1, h2 = st.columns([3,1])
with h1:
    dmg_num = hist["Damage"].map({"High":3,"Medium":2,"Low":1})
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(
        x=hist["Date"], y=hist["Mag"], name="Magnitude",
        line=dict(color="#1e90ff",width=2), mode="lines+markers",
        marker=dict(size=7, color=["#ff3b4e" if d==3 else "#ff8c00" if d==2 else "#1dd176"
                                    for d in dmg_num]),
    ))
    fig_h.add_trace(go.Bar(
        x=hist["Date"], y=dmg_num, name="Damage Level",
        marker_color=["#ff3b4e" if d=="High" else "#ff8c00" if d=="Medium" else "#1dd176"
                      for d in hist["Damage"]],
        opacity=0.35, yaxis="y2",
    ))
    fig_h.update_layout(
        height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#4a7a9b",showgrid=False),
        yaxis=dict(color="#4a7a9b",showgrid=False,title="Magnitude",titlefont=dict(size=10)),
        yaxis2=dict(overlaying="y",side="right",showticklabels=False),
        legend=dict(font=dict(color="#4a7a9b",size=10)),
        margin=dict(l=0,r=0,t=0,b=0),
    )
    st.plotly_chart(fig_h, use_container_width=True)

with h2:
    avg = hist["Damage"].map({"High":3,"Medium":2,"Low":1}).mean()
    avg_lbl = "High" if avg>=2.5 else "Medium" if avg>=1.5 else "Low"
    for (lbl, val, clr) in [
        ("Avg Damage",     avg_lbl,                   severity_color(avg_lbl)),
        ("High Events",    f"{(hist['Damage']=='High').sum()} / 10", "c-red"),
        ("Peak Magnitude", f"M {hist['Mag'].max():.1f}", "c-orange"),
    ]:
        st.markdown(f"""<div class="kpi-card" style="margin-bottom:8px">
          <div class="kpi-label">{lbl}</div>
          <div class="kpi-value {clr}">{val}</div>
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# AUTO-ADVANCE SIMULATION
# ──────────────────────────────────────────
if st.session_state.sim_running:
    import time
    time.sleep(0.7)
    st.rerun()

# ──────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────
st.markdown("""
<div style='text-align:center;margin-top:40px;padding:16px;
     border-top:1px solid #172d4a;color:#1e3a5a;font-size:11px;
     font-family:Share Tech Mono,monospace;letter-spacing:2px'>
  QUAKEGUARD v3 · ML-POWERED EARTHQUAKE RESPONSE SYSTEM · ACADEMIC DEMONSTRATION
</div>
""", unsafe_allow_html=True)

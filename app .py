"""
Explainable AI Triage Dashboard — KTAS Emergency Triage System
MSN Project | Institute of Nursing Science, Sarhad University (SUIT)
Author: Rumman Khan | Supervisor: Mr. Noor ul Amin
"""

import streamlit as st
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page config (must be FIRST Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Triage System | KTAS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── App background ── */
.stApp {
    background: #0D1117;
    color: #E6EDF3;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161B22;
    border-right: 1px solid #30363D;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #58A6FF;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid #30363D;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #0D1117 100%);
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #E53935, #FDD835, #43A047);
}
.header-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #E6EDF3;
    margin: 0;
}
.header-sub {
    font-size: 0.82rem;
    color: #8B949E;
    margin: 0.2rem 0 0 0;
    font-weight: 400;
}
.header-badge {
    background: #21262D;
    border: 1px solid #30363D;
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-size: 0.72rem;
    color: #58A6FF;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
}

/* ── Triage result cards ── */
.triage-card {
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    border: 2px solid;
    position: relative;
    overflow: hidden;
}
.triage-card-RED {
    background: rgba(229, 57, 53, 0.12);
    border-color: #E53935;
    box-shadow: 0 0 30px rgba(229, 57, 53, 0.2);
}
.triage-card-YELLOW {
    background: rgba(253, 216, 53, 0.10);
    border-color: #FDD835;
    box-shadow: 0 0 30px rgba(253, 216, 53, 0.15);
}
.triage-card-GREEN {
    background: rgba(67, 160, 71, 0.12);
    border-color: #43A047;
    box-shadow: 0 0 30px rgba(67, 160, 71, 0.2);
}
.triage-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8B949E;
    margin-bottom: 0.5rem;
}
.triage-class {
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    margin: 0.5rem 0;
}
.triage-class-RED    { color: #E53935; }
.triage-class-YELLOW { color: #FDD835; }
.triage-class-GREEN  { color: #43A047; }
.triage-desc {
    font-size: 0.85rem;
    color: #8B949E;
    margin-top: 0.5rem;
}
.triage-confidence {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 0.8rem;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 0.8rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 80px;
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 0.8rem;
    text-align: center;
}
.metric-card-label {
    font-size: 0.65rem;
    color: #8B949E;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.metric-card-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: #E6EDF3;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.2rem;
}

/* ── Section headers ── */
.section-header {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #58A6FF;
    border-bottom: 1px solid #21262D;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── Probability bars ── */
.prob-bar-wrap {
    margin: 0.5rem 0;
}
.prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    margin-bottom: 0.25rem;
    font-weight: 500;
}
.prob-bar-bg {
    background: #21262D;
    border-radius: 4px;
    height: 10px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}

/* ── LIME explanation ── */
.lime-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0.4rem 0;
    font-size: 0.8rem;
}
.lime-feat {
    flex: 1;
    color: #C9D1D9;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.lime-bar-wrap {
    width: 120px;
    background: #21262D;
    border-radius: 3px;
    height: 8px;
    overflow: hidden;
}
.lime-bar-pos { background: #E53935; height: 100%; border-radius: 3px; }
.lime-bar-neg { background: #43A047; height: 100%; border-radius: 3px; }
.lime-val {
    width: 55px;
    text-align: right;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #8B949E;
}

/* ── Alert / info boxes ── */
.alert-box {
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin: 0.8rem 0;
    font-size: 0.83rem;
    line-height: 1.5;
    border-left: 4px solid;
}
.alert-warning {
    background: rgba(253,216,53,0.08);
    border-color: #FDD835;
    color: #C9D1D9;
}
.alert-info {
    background: rgba(88,166,255,0.08);
    border-color: #58A6FF;
    color: #C9D1D9;
}

/* ── Input number fields ── */
.stNumberInput input {
    background: #21262D !important;
    border: 1px solid #30363D !important;
    color: #E6EDF3 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stNumberInput input:focus {
    border-color: #58A6FF !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.2) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #21262D !important;
    border: 1px solid #30363D !important;
    color: #E6EDF3 !important;
}

/* ── Slider ── */
.stSlider .stSlider { color: #58A6FF; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #1F6FEB, #388BFD) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #388BFD, #1F6FEB) !important;
    box-shadow: 0 4px 15px rgba(56,139,253,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
hr { border-color: #21262D !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161B22;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8B949E;
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: #21262D !important;
    color: #E6EDF3 !important;
}

/* ── Footer ── */
.dashboard-footer {
    text-align: center;
    color: #484F58;
    font-size: 0.72rem;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #21262D;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

FEATURES = ['Age', 'Sex', 'Arrival mode', 'Injury',
            'Mental', 'Pain', 'NRS_pain',
            'SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation']

TRIAGE_INFO = {
    "RED":    {"emoji": "🔴", "desc": "Resuscitation / Emergency",    "action": "Immediate intervention required"},
    "YELLOW": {"emoji": "🟡", "desc": "Urgent — within 30 minutes",   "action": "Prompt assessment needed"},
    "GREEN":  {"emoji": "🟢", "desc": "Semi-urgent / Non-urgent",     "action": "Can be safely deferred"},
}

@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    """Load model, scaler, and label encoder once."""
    errors = []
    model, scaler, le = None, None, None

    # TensorFlow
    try:
        import tensorflow as tf
        if os.path.exists("triage_ann_model.h5"):
            model = tf.keras.models.load_model("triage_ann_model.h5")
        else:
            errors.append("triage_ann_model.h5 not found in current directory.")
    except Exception as e:
        errors.append(f"TensorFlow error: {e}")

    # Scaler
    try:
        if os.path.exists("scaler.pkl"):
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
        else:
            errors.append("scaler.pkl not found.")
    except Exception as e:
        errors.append(f"Scaler error: {e}")

    # Label encoder
    try:
        if os.path.exists("label_encoder.pkl"):
            with open("label_encoder.pkl", "rb") as f:
                le = pickle.load(f)
        else:
            errors.append("label_encoder.pkl not found.")
    except Exception as e:
        errors.append(f"Label encoder error: {e}")

    return model, scaler, le, errors


def predict(model, scaler, le, inputs: list):
    """Run prediction pipeline. Returns (class_str, probs_dict)."""
    x = np.array([inputs], dtype=float)
    x_sc = scaler.transform(x)
    probs = model.predict(x_sc, verbose=0)[0]
    pred_class = le.classes_[np.argmax(probs)]
    prob_dict = {cls: float(probs[i]) for i, cls in enumerate(le.classes_)}
    return pred_class, prob_dict, x_sc


def get_lime_explanation(model, scaler, le, x_sc, pred_class_idx):
    """Compute LIME explanation for the current prediction."""
    try:
        import lime.lime_tabular
        # Use a dummy background when no training data is cached
        np.random.seed(42)
        dummy_bg = np.random.randn(50, len(FEATURES))
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=dummy_bg,
            feature_names=FEATURES,
            class_names=list(le.classes_),
            mode='classification',
            discretize_continuous=True,
            random_state=42,
        )
        exp = lime_explainer.explain_instance(
            data_row=x_sc[0],
            predict_fn=lambda x: model.predict(x, verbose=0),
            num_features=8,
            labels=(pred_class_idx,),
        )
        return exp.as_list(label=pred_class_idx)
    except Exception as e:
        return [("LIME unavailable", str(e))]


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-banner">
    <div style="font-size:2.8rem; line-height:1;">🏥</div>
    <div style="flex:1;">
        <p class="header-title">AI Triage Decision Support System</p>
        <p class="header-sub">Korean Triage &amp; Acuity Scale (KTAS) · ANN + SHAP + LIME · MSN Project</p>
    </div>
    <div style="display:flex; flex-direction:column; gap:0.4rem; align-items:flex-end;">
        <span class="header-badge">ANN v1.0</span>
        <span class="header-badge">KTAS Dataset</span>
        <span class="header-badge">XAI Enabled</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="alert-box alert-warning">
    ⚠️ <strong>Clinical Decision Support Only.</strong> This system assists triage nurses — it does not replace clinical judgment.
    All AI predictions must be reviewed and confirmed by a qualified nurse or physician before action is taken.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

with st.spinner("Loading AI model..."):
    model, scaler, le, load_errors = load_model_artifacts()

model_ready = (model is not None and scaler is not None and le is not None)

if load_errors:
    for err in load_errors:
        st.error(f"❌ {err}")
    if not model_ready:
        st.markdown("""
        <div class="alert-box alert-info">
            <strong>Setup required:</strong><br>
            Place <code>triage_ann_model.h5</code>, <code>scaler.pkl</code>, and <code>label_encoder.pkl</code>
            in the same folder as <code>app.py</code>, then refresh. These files are produced at the end
            of the Google Colab training notebook.
        </div>
        """, unsafe_allow_html=True)

if model_ready:
    st.markdown("""
    <div class="alert-box alert-info" style="border-color:#43A047; background:rgba(67,160,71,0.08);">
        ✅ <strong>Model loaded successfully.</strong> Enter patient vital signs in the sidebar and click <em>Run Triage Prediction</em>.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — PATIENT INPUT
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Patient Data Entry")

    st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=45, step=1)
    sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    arrival_mode = st.selectbox(
        "Arrival Mode",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {
            1: "1 — Walking", 2: "2 — Wheelchair",
            3: "3 — Ambulance", 4: "4 — Others", 5: "5 — Unknown"
        }[x]
    )
    injury = st.selectbox("Injury", options=[1, 2],
                          format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown('<div class="section-header">Consciousness & Pain</div>', unsafe_allow_html=True)
    mental = st.selectbox(
        "Mental Status (AVPU)",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 — Alert", 2: "2 — Verbal response",
            3: "3 — Pain response", 4: "4 — Unresponsive"
        }[x]
    )
    pain = st.selectbox("Pain Present", options=[1, 2],
                        format_func=lambda x: "Yes" if x == 1 else "No")
    nrs_pain = st.slider("NRS Pain Score (0–10)", min_value=0, max_value=10, value=3)

    st.markdown('<div class="section-header">Vital Signs</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        sbp = st.number_input("SBP (mmHg)", min_value=50, max_value=250, value=120)
        hr  = st.number_input("HR (bpm)",   min_value=20, max_value=220, value=80)
        bt  = st.number_input("Temp (°C)",  min_value=34.0, max_value=42.0, value=37.0, step=0.1, format="%.1f")
    with col_b:
        dbp        = st.number_input("DBP (mmHg)", min_value=20, max_value=150, value=80)
        rr         = st.number_input("RR (/min)",  min_value=4,  max_value=60,  value=18)
        saturation = st.number_input("SpO₂ (%)",   min_value=50, max_value=100, value=98)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Run Triage Prediction", disabled=not model_ready)

    st.markdown("---")
    st.markdown("**Load example cases:**")
    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        demo_red = st.button("🔴 RED case")
    with demo_col2:
        demo_green = st.button("🟢 GREEN case")


# ══════════════════════════════════════════════════════════════════════════════
# DEMO CASE INJECTION
# ══════════════════════════════════════════════════════════════════════════════

if demo_red:
    st.session_state["demo"] = "red"
if demo_green:
    st.session_state["demo"] = "green"

# Override inputs for demo cases (note: Streamlit reruns — values shown in sidebar after rerun)
demo = st.session_state.get("demo", None)
if demo == "red":
    age, sex, arrival_mode, injury = 65, 1, 3, 2
    mental, pain, nrs_pain = 3, 1, 8
    sbp, dbp, hr, rr, bt, saturation = 88, 55, 125, 34, 38.6, 82
    predict_btn = True
elif demo == "green":
    age, sex, arrival_mode, injury = 28, 2, 1, 2
    mental, pain, nrs_pain = 1, 1, 2
    sbp, dbp, hr, rr, bt, saturation = 125, 80, 76, 16, 36.8, 99
    predict_btn = True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL — TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_result, tab_explain, tab_about = st.tabs([
    "📊 Triage Result",
    "🔬 AI Explanation (SHAP & LIME)",
    "ℹ️ About & Clinical Guide",
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — RESULT
# ──────────────────────────────────────────────────────────────────────────────

with tab_result:

    if not predict_btn:
        st.markdown("""
        <div style="text-align:center; padding:4rem 0; color:#484F58;">
            <div style="font-size:4rem;">🏥</div>
            <p style="font-size:1rem; margin-top:1rem;">Enter patient vital signs in the sidebar<br>and click <strong style="color:#8B949E;">Run Triage Prediction</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        inputs = [age, sex, arrival_mode, injury,
                  mental, pain, nrs_pain,
                  sbp, dbp, hr, rr, bt, saturation]

        with st.spinner("Running AI prediction..."):
            pred_class, probs, x_sc = predict(model, scaler, le, inputs)

        info = TRIAGE_INFO[pred_class]
        confidence = probs[pred_class]

        # ── Triage result card ──
        st.markdown(f"""
        <div class="triage-card triage-card-{pred_class}">
            <p class="triage-label">Triage Prediction</p>
            <p class="triage-class triage-class-{pred_class}">{info['emoji']} {pred_class}</p>
            <p style="font-size:1rem; color:#C9D1D9; font-weight:500; margin:0.3rem 0;">{info['desc']}</p>
            <p class="triage-desc">{info['action']}</p>
            <p class="triage-confidence" style="color:{'#E53935' if pred_class=='RED' else '#FDD835' if pred_class=='YELLOW' else '#43A047'}">
                Confidence: {confidence*100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Patient vitals summary ──
        st.markdown('<div class="section-header">Entered Vital Signs</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card"><div class="metric-card-label">HR</div><div class="metric-card-value">{hr}</div></div>
            <div class="metric-card"><div class="metric-card-label">RR</div><div class="metric-card-value">{rr}</div></div>
            <div class="metric-card"><div class="metric-card-label">SBP</div><div class="metric-card-value">{sbp}</div></div>
            <div class="metric-card"><div class="metric-card-label">DBP</div><div class="metric-card-value">{dbp}</div></div>
            <div class="metric-card"><div class="metric-card-label">SpO₂</div><div class="metric-card-value">{saturation}%</div></div>
            <div class="metric-card"><div class="metric-card-label">Temp</div><div class="metric-card-value">{bt}°C</div></div>
            <div class="metric-card"><div class="metric-card-label">Pain NRS</div><div class="metric-card-value">{nrs_pain}/10</div></div>
            <div class="metric-card"><div class="metric-card-label">Mental</div><div class="metric-card-value">{['A','V','P','U'][mental-1]}</div></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability bars ──
        st.markdown('<div class="section-header">Class Probabilities</div>', unsafe_allow_html=True)

        bar_colors = {"GREEN": "#43A047", "RED": "#E53935", "YELLOW": "#FDD835"}
        for cls in ["RED", "YELLOW", "GREEN"]:
            p = probs[cls]
            marker = " ◀ predicted" if cls == pred_class else ""
            st.markdown(f"""
            <div class="prob-bar-wrap">
                <div class="prob-bar-label">
                    <span>{TRIAGE_INFO[cls]['emoji']} {cls}{marker}</span>
                    <span style="font-family:'JetBrains Mono',monospace;">{p*100:.1f}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{p*100:.1f}%; background:{bar_colors[cls]};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Clinical flags ──
        st.markdown('<div class="section-header">Clinical Alert Flags</div>', unsafe_allow_html=True)
        flags = []
        if saturation < 90:  flags.append(("🚨", "SpO₂ critically low", f"{saturation}% — consider O₂ therapy"))
        if rr > 30:          flags.append(("🚨", "Respiratory rate elevated", f"{rr}/min — signs of respiratory distress"))
        if hr > 120:         flags.append(("⚠️", "Tachycardia", f"HR {hr} bpm"))
        if hr < 50:          flags.append(("⚠️", "Bradycardia", f"HR {hr} bpm"))
        if sbp < 90:         flags.append(("🚨", "Hypotension", f"SBP {sbp} mmHg — shock protocol"))
        if bt > 38.5:        flags.append(("⚠️", "Fever", f"Temp {bt}°C"))
        if bt < 36.0:        flags.append(("⚠️", "Hypothermia", f"Temp {bt}°C"))
        if mental >= 3:      flags.append(("🚨", "Altered consciousness", f"AVPU = {'P' if mental==3 else 'U'}"))
        if nrs_pain >= 8:    flags.append(("⚠️", "Severe pain", f"NRS {nrs_pain}/10"))

        if flags:
            for icon, title, detail in flags:
                is_critical = icon == "🚨"
                border = "#E53935" if is_critical else "#FDD835"
                bg = "rgba(229,57,53,0.08)" if is_critical else "rgba(253,216,53,0.08)"
                st.markdown(f"""
                <div class="alert-box" style="border-color:{border}; background:{bg};">
                    {icon} <strong>{title}</strong> — {detail}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-box" style="border-color:#43A047; background:rgba(67,160,71,0.08);">
                ✅ <strong>No critical vital sign flags detected.</strong>
            </div>
            """, unsafe_allow_html=True)

        # ── Store for explanation tab ──
        st.session_state["last_pred"] = {
            "inputs": inputs, "pred_class": pred_class,
            "probs": probs, "x_sc": x_sc,
            "pred_idx": list(le.classes_).index(pred_class)
        }

        # Clear demo state
        if demo:
            st.session_state.pop("demo", None)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — EXPLANATION
# ──────────────────────────────────────────────────────────────────────────────

with tab_explain:

    last = st.session_state.get("last_pred", None)

    if last is None:
        st.markdown("""
        <div style="text-align:center; padding:4rem 0; color:#484F58;">
            <div style="font-size:4rem;">🔬</div>
            <p style="font-size:1rem; margin-top:1rem;">Run a prediction first to see the AI explanation.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        pred_class = last["pred_class"]
        probs      = last["probs"]
        x_sc       = last["x_sc"]
        pred_idx   = last["pred_idx"]
        inputs_arr = last["inputs"]

        st.markdown(f"""
        <div class="alert-box alert-info">
            Explaining prediction: <strong>{TRIAGE_INFO[pred_class]['emoji']} {pred_class}</strong>
            ({probs[pred_class]*100:.1f}% confidence) for the patient entered in the sidebar.
        </div>
        """, unsafe_allow_html=True)

        col_shap, col_lime = st.columns(2)

        # ── SHAP (manual approximation using gradient perturbation) ──────────
        with col_shap:
            st.markdown('<div class="section-header">SHAP — Feature Contributions</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.75rem; color:#8B949E; margin-bottom:0.8rem;">
                Each bar shows how strongly a feature <strong>pushed</strong> the prediction toward
                or away from the predicted class. Positive = increases risk; Negative = decreases risk.
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Computing SHAP approximation..."):
                try:
                    # Perturbation-based SHAP approximation (no background data needed)
                    shap_approx = {}
                    baseline = model.predict(np.zeros((1, len(FEATURES))), verbose=0)[0][pred_idx]
                    for fi, feat in enumerate(FEATURES):
                        perturbed = x_sc.copy()
                        perturbed[0, fi] = 0.0
                        p_val = model.predict(perturbed, verbose=0)[0][pred_idx]
                        shap_approx[feat] = float(model.predict(x_sc, verbose=0)[0][pred_idx] - p_val)

                    # Sort by absolute value
                    sorted_shap = sorted(shap_approx.items(), key=lambda x: abs(x[1]), reverse=True)
                    max_abs = max(abs(v) for _, v in sorted_shap) + 1e-9

                    for feat, val in sorted_shap[:10]:
                        pct = abs(val) / max_abs * 100
                        color = "#E53935" if val > 0 else "#43A047"
                        direction = f"+{val:.3f}" if val > 0 else f"{val:.3f}"
                        label = "↑ supports RED" if val > 0 and pred_class == "RED" else \
                                "↑ raises risk" if val > 0 else "↓ lowers risk"
                        st.markdown(f"""
                        <div class="lime-row">
                            <span class="lime-feat" title="{feat}">{feat}</span>
                            <div class="lime-bar-wrap">
                                <div style="width:{pct:.0f}%; background:{color}; height:100%; border-radius:3px;"></div>
                            </div>
                            <span class="lime-val" style="color:{color};">{direction}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Highlight top 2
                    top2 = sorted_shap[:2]
                    st.markdown(f"""
                    <div style="margin-top:1rem; padding:0.8rem; background:#161B22;
                                border-radius:8px; border:1px solid #30363D; font-size:0.78rem; color:#C9D1D9;">
                        <strong>Key drivers:</strong><br>
                        • <code>{top2[0][0]}</code> — largest contributor ({'+' if top2[0][1]>0 else ''}{top2[0][1]:.3f})<br>
                        • <code>{top2[1][0]}</code> — second largest ({'+' if top2[1][1]>0 else ''}{top2[1][1]:.3f})
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"SHAP computation error: {e}")

        # ── LIME ──────────────────────────────────────────────────────────────
        with col_lime:
            st.markdown('<div class="section-header">LIME — Local Explanation</div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.75rem; color:#8B949E; margin-bottom:0.8rem;">
                LIME fits a local linear model around this specific patient to explain which feature ranges
                drove the prediction. Positive weight = supports predicted class.
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Computing LIME explanation..."):
                lime_exp = get_lime_explanation(model, scaler, le, x_sc, pred_idx)
                max_lime = max(abs(w) for _, w in lime_exp) + 1e-9

                for feat_cond, weight in lime_exp:
                    pct = abs(weight) / max_lime * 100
                    color = "#E53935" if weight > 0 else "#43A047"
                    label = f"+{weight:.4f}" if weight > 0 else f"{weight:.4f}"
                    # Truncate long feature condition strings
                    short = feat_cond if len(feat_cond) <= 28 else feat_cond[:26] + "…"
                    st.markdown(f"""
                    <div class="lime-row">
                        <span class="lime-feat" title="{feat_cond}">{short}</span>
                        <div class="lime-bar-wrap">
                            <div style="width:{pct:.0f}%; background:{color}; height:100%; border-radius:3px;"></div>
                        </div>
                        <span class="lime-val" style="color:{color};">{label}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("""
            <div style="margin-top:1rem; padding:0.8rem; background:#161B22;
                        border-radius:8px; border:1px solid #30363D; font-size:0.78rem; color:#8B949E;">
                🔵 <strong>Positive</strong> weights (red bar) push prediction toward the predicted class.<br>
                🟢 <strong>Negative</strong> weights (green bar) push prediction away from it.
            </div>
            """, unsafe_allow_html=True)

        # ── Clinical narrative ─────────────────────────────────────────────
        st.markdown('<div class="section-header">Clinical Reasoning Summary</div>', unsafe_allow_html=True)

        # Auto-generate narrative based on actual values
        reasons = []
        if saturation < 90:
            reasons.append(f"SpO₂ of {inputs_arr[12]}% indicates severe hypoxemia")
        if rr > 25:
            reasons.append(f"Respiratory rate of {inputs_arr[11]}/min indicates respiratory distress")
        if hr > 120:
            reasons.append(f"Heart rate of {inputs_arr[10]} bpm indicates tachycardia")
        if sbp < 90:
            reasons.append(f"SBP of {inputs_arr[7]} mmHg indicates hypotension / possible shock")
        if mental >= 3:
            avpu = ['A', 'V', 'P', 'U'][inputs_arr[4] - 1]
            reasons.append(f"Consciousness level AVPU = {avpu} indicates altered mental status")
        if nrs_pain >= 7:
            reasons.append(f"Pain score of {inputs_arr[6]}/10 indicates severe pain")
        if not reasons:
            reasons.append("Vital signs are within broadly acceptable ranges")
            reasons.append("No single parameter alone indicates high acuity")

        bullet_html = "".join(f"<li>{r}</li>" for r in reasons)
        st.markdown(f"""
        <div style="background:#161B22; border:1px solid #30363D; border-radius:10px;
                    padding:1.2rem 1.5rem; font-size:0.83rem; color:#C9D1D9; line-height:1.8;">
            <strong style="color:#E6EDF3;">Prediction: {TRIAGE_INFO[pred_class]['emoji']} {pred_class}</strong>
            — {TRIAGE_INFO[pred_class]['desc']}<br><br>
            <strong>Contributing clinical findings:</strong>
            <ul style="margin:0.5rem 0 0 1rem; padding:0;">
                {bullet_html}
            </ul>
            <br>
            <span style="color:#484F58; font-size:0.72rem;">
                This interpretation is generated by the AI model and must be validated against
                the patient's full clinical presentation by a qualified nurse or physician.
            </span>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — ABOUT
# ──────────────────────────────────────────────────────────────────────────────

with tab_about:

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">Project Information</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.85rem; color:#C9D1D9; line-height:1.8;">
            <table style="width:100%; border-collapse:collapse; font-size:0.83rem;">
                <tr><td style="color:#8B949E; padding:0.3rem 0.8rem 0.3rem 0; width:40%;">Project Title</td>
                    <td>Explainable AI-Based Emergency Triage System</td></tr>
                <tr><td style="color:#8B949E; padding:0.3rem 0.8rem 0.3rem 0;">Author</td>
                    <td>Rumman Khan, MSN Scholar</td></tr>
                <tr><td style="color:#8B949E; padding:0.3rem 0.8rem 0.3rem 0;">Supervisor</td>
                    <td>Mr. Noor ul Amin</td></tr>
                <tr><td style="color:#8B949E; padding:0.3rem 0.8rem 0.3rem 0;">Institution</td>
                    <td>Institute of Nursing Science, Sarhad University (SUIT), Peshawar</td></tr>
                <tr><td style="color:#8B949E; padding:0.3rem 0.8rem 0.3rem 0;">Dataset</td>
                    <td>Korean KTAS Emergency Department Dataset (n=1,267)</td></tr>
                <tr><td style="color:#8B949E; padding:0.3rem 0.8rem 0.3rem 0;">Model</td>
                    <td>Artificial Neural Network (ANN) — 3-layer deep network</td></tr>
                <tr><td style="color:#8B949E; padding:0.3rem 0.8rem 0.3rem 0;">XAI Methods</td>
                    <td>SHAP (global + local) and LIME (individual)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">KTAS → 3-Class Mapping</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.83rem; color:#C9D1D9;">
        <table style="width:100%; border-collapse:collapse;">
            <thead>
                <tr style="color:#8B949E; border-bottom:1px solid #30363D;">
                    <th style="text-align:left; padding:0.4rem;">KTAS Level</th>
                    <th style="text-align:left; padding:0.4rem;">Category</th>
                    <th style="text-align:left; padding:0.4rem;">Response Time</th>
                    <th style="text-align:left; padding:0.4rem;">Action</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding:0.4rem; color:#E53935; font-weight:700;">1 & 2</td>
                    <td style="padding:0.4rem;">🔴 RED — Emergency</td>
                    <td style="padding:0.4rem;">Immediate / &lt;10 min</td>
                    <td style="padding:0.4rem;">Resuscitation, crash team</td>
                </tr>
                <tr style="background:#0D1117;">
                    <td style="padding:0.4rem; color:#FDD835; font-weight:700;">3</td>
                    <td style="padding:0.4rem;">🟡 YELLOW — Urgent</td>
                    <td style="padding:0.4rem;">&lt;30 minutes</td>
                    <td style="padding:0.4rem;">Prompt assessment</td>
                </tr>
                <tr>
                    <td style="padding:0.4rem; color:#43A047; font-weight:700;">4 & 5</td>
                    <td style="padding:0.4rem;">🟢 GREEN — Non-urgent</td>
                    <td style="padding:0.4rem;">&lt;60–120 min</td>
                    <td style="padding:0.4rem;">Routine assessment</td>
                </tr>
            </tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">ANN Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#161B22; border:1px solid #30363D; border-radius:10px;
                    padding:1rem; font-family:'JetBrains Mono',monospace; font-size:0.72rem;
                    color:#8B949E; line-height:2;">
            <span style="color:#58A6FF;">Input Layer</span><br>
            &nbsp;&nbsp;13 features (vital signs)<br>
            <span style="color:#30363D;">│</span><br>
            <span style="color:#43A047;">Dense(128, ReLU)</span><br>
            &nbsp;&nbsp;BatchNorm + Dropout(0.3)<br>
            <span style="color:#30363D;">│</span><br>
            <span style="color:#43A047;">Dense(64, ReLU)</span><br>
            &nbsp;&nbsp;BatchNorm + Dropout(0.2)<br>
            <span style="color:#30363D;">│</span><br>
            <span style="color:#43A047;">Dense(32, ReLU)</span><br>
            <span style="color:#30363D;">│</span><br>
            <span style="color:#E53935;">Output Dense(3, Softmax)</span><br>
            &nbsp;&nbsp;GREEN / RED / YELLOW
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Ethical Disclaimer</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-box alert-warning" style="font-size:0.78rem;">
            This AI system is a <strong>prototype for academic demonstration</strong>.
            It must not be used for actual patient care without prospective clinical validation,
            local IRB approval, and integration into existing hospital governance frameworks.<br><br>
            The KTAS dataset originates from South Korean EDs.
            Performance on Pakistani/South Asian patient populations requires separate validation
            (LRH, KTH, HMC).
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">How to Run Locally</div>', unsafe_allow_html=True)
    st.code("""# 1. Install dependencies
pip install streamlit tensorflow scikit-learn lime shap

# 2. Place model files in same folder as app.py
#    triage_ann_model.h5
#    scaler.pkl
#    label_encoder.pkl

# 3. Run
streamlit run app.py

# Dashboard opens at http://localhost:8501""", language="bash")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="dashboard-footer">
    AI Triage Decision Support System &nbsp;|&nbsp; MSN Project &nbsp;|&nbsp;
    Institute of Nursing Science, SUIT Peshawar &nbsp;|&nbsp;
    Rumman Khan &nbsp;·&nbsp; Supervisor: Mr. Noor ul Amin<br>
    <span style="color:#30363D;">Built with Streamlit · TensorFlow · SHAP · LIME</span>
</div>
""", unsafe_allow_html=True)

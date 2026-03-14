# Acceuil.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration globale ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Bias Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Chargement des données ───────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "healthcare-dataset-stroke-data.csv"

@st.cache_data
def load_data(path: Path):
    df = pd.read_csv(path)
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df = df[df["gender"] != "Other"].copy()
    return df

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"📁 Fichier introuvable : `{DATA_PATH.name}`.\n\n"
        "Vérifie qu'il est présent dans ton projet."
    )
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Healthcare Bias Analysis")
    st.markdown("---")
    st.caption("Dataset : Stroke Prediction | Kaggle")
    st.caption("Parcours A — Mastère 2 Data & IA")

# ── Fond blanc + Hero bleu (propre) ──────────────────────────────────────────
st.markdown("""
<style>
/* Fond global blanc */
[data-testid="stAppViewContainer"]{
    background: #ffffff;
    color: #000000;
}

/* Sidebar blanche */
[data-testid="stSidebar"]{
    background: #f8f9fa;
}

/* Hero */
.hero{
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 50%, #1e40af 100%);
  color: white;
  text-align: center;
  border-radius: 24px;
  padding: 1.5rem 1rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 6px 25px rgba(0,0,0,0.15);
}

/* Titres */
h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
}

/* Texte */
p, .stMarkdown {
    color: #1e293b !important;
}

/* KPIs */
[data-testid="stMetric"]{
  background: #ffffff;
  padding: 1rem;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
[data-testid="stMetric"] > label{
    color: #475569 !important;
}
[data-testid="stMetric"] > div > div{
    color: #0f172a !important;
}

/* DataFrame blanc */
.dataframe th, .dataframe td {
    background: #ffffff !important;
    color: #0f172a !important;
    border-color: #e2e8f0 !important;
}
</style>

<div class="hero">
    <h1 style="font-size:3rem; margin:0;">Healthcare Bias Analysis</h1>
</div>
""", unsafe_allow_html=True)

# ── KPIs ─────────────────────────────────────────────────────────────────────
total = len(df)
n_stroke = int(df["stroke"].sum())
stroke_rate = n_stroke / total * 100 if total else 0
missing_bmi = int(df["bmi"].isnull().sum())
missing_pct = missing_bmi / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", f"{total:,}")
c2.metric("Cas d'AVC", f"{n_stroke}", f"{stroke_rate:.1f}%")
c3.metric("Variables", f"{df.shape[1]}")
c4.metric("Valeurs manquantes (BMI)", f"{missing_bmi}", f"{missing_pct:.1f}%")

st.markdown("---")

# ── Contexte ─────────────────────────────────────────────────────────────────
st.subheader("Contexte et objectifs")
st.markdown("""
L'**AVC (Accident Vasculaire Cérébral)** est la 2ème cause de mortalité mondiale et la 3ème
cause de handicap.

Cette application vise à :

- **Explorer les données médicales**
- **Prédire le risque d’AVC**
- **Analyser les biais potentiels** du modèle
- Identifier les influences du **genre** et de la **résidence** (Rural/Urban)

L’objectif : améliorer l’équité et la fiabilité des prédictions.
""")

st.markdown("---")

# ── Aperçu des données ──────────────────────────────────────────────────────
st.subheader("Aperçu des données")

st.dataframe(df.head(10), width='stretch')
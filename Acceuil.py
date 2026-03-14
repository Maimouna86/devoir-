import streamlit as st
import pandas as pd
import numpy as np

# ── Configuration globale ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Bias Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Chargement des données (cached) ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df = df[df['gender'] != 'Other'].copy()
    return df

df = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Healthcare Bias Analysis")
    st.markdown("---")
    st.caption("Dataset : Stroke Prediction | Kaggle")
    st.caption("Parcours A — Mastère 2 Data & IA")

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);}
.stMetric > label {font-size: 1.1rem !important;}
.stMetric > div > div {font-size: 2.5rem !important; font-weight: 700;}
.section-card {background: white; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); padding: 2rem; margin: 1rem 0;}
.btn-primary {background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important; border-radius: 12px !important;}
</style>
<div class="section-card" style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 50%, #1e40af 100%); color: white; text-align: center;">
<h1 style="font-size: 4rem; margin: 0 0 1rem 0; text-shadow: 0 4px 10px rgba(0,0,0,0.3);">Healthcare Bias Analysis</h1>
</div>
""", unsafe_allow_html=True)

# ── KPIs ─────────────────────────────────────────────────────────────────────
total = len(df)
n_stroke = df['stroke'].sum()
stroke_rate = n_stroke / total * 100
missing_bmi = df['bmi'].isnull().sum()
missing_pct = missing_bmi / total * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", f"{total:,}", help="Nombre total de lignes dans le dataset")
c2.metric("Cas d'AVC", f"{n_stroke}", f"{stroke_rate:.1f}% du total")
c3.metric("Variables", f"{df.shape[1]}", help="Nombre de colonnes")
c4.metric("Valeurs manquantes", f"{missing_bmi}", f"{missing_pct:.1f}% (BMI uniquement)")

st.markdown("---")

# ── Contexte ─────────────────────────────────────────────────────────────────
col = st.columns(1)[0]

with col:
    st.subheader("Contexte et objectifs")
    st.markdown("""
    L'**AVC (Accident Vasculaire Cérébral)** est la 2ème cause de mortalité mondiale et la 3ème
    cause de handicap. 

    Notre objectif est double : **prédire le risque d'AVC** et **détecter les biais potentiels**
    liés au **genre** (Male/Female) et à la **zone géographique** (Rural/Urban). Ces biais,
    s'ils existent dans les données d'entraînement, pourraient conduire un modèle à traiter
    inégalement différents groupes de population — avec des conséquences médicales réelles.
    """)

st.markdown("---")

# ── Aperçu données ───────────────────────────────────────────────────────────
st.subheader("Dataset : aperçu des données")
st.dataframe(df.head(10), width='stretch')
import streamlit as st
import pandas as pd
import numpy as np

# ── Configuration globale ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Bias Detector",
    page_icon="🧠",
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
    st.image("https://img.icons8.com/color/96/brain.png", width=80)
    st.title("🧠 Stroke Bias Detector")
    st.markdown("---")
    st.caption("Dataset : Stroke Prediction | Kaggle")
    st.caption("Parcours A — Mastère 2 Data & IA")

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            padding: 3rem 2rem; border-radius: 16px; margin-bottom: 2rem; text-align:center;">
    <h1 style="color:#e94560; font-size:3rem; margin:0;">🧠 Stroke Bias Detector</h1>
    <p style="color:#a8b2d8; font-size:1.2rem; margin-top:0.8rem;">
        Analyse des biais dans la prédiction du risque d'AVC
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPIs ─────────────────────────────────────────────────────────────────────
total = len(df)
n_stroke = df['stroke'].sum()
stroke_rate = n_stroke / total * 100
missing_bmi = df['bmi'].isnull().sum()
missing_pct = missing_bmi / total * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("👥 Patients", f"{total:,}", help="Nombre total de lignes dans le dataset")
c2.metric("🧠 Cas d'AVC", f"{n_stroke}", f"{stroke_rate:.1f}% du total")
c3.metric("📋 Variables", f"{df.shape[1]}", help="Nombre de colonnes")
c4.metric("❓ Valeurs manquantes", f"{missing_bmi}", f"{missing_pct:.1f}% (BMI uniquement)")

st.markdown("---")

# ── Contexte ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📌 Contexte et Problématique")
    st.markdown("""
    L'**AVC (Accident Vasculaire Cérébral)** est la 2ème cause de mortalité mondiale et la 3ème
    cause de handicap. Chaque année, environ 15 millions de personnes en sont victimes.
    La détection précoce des patients à risque est donc un enjeu de santé publique majeur.

    Ce dataset, issu de **Kaggle**, regroupe les données médicales de **5 109 patients** avec
    12 variables cliniques et démographiques : âge, hypertension, maladie cardiaque, IMC,
    niveau de glucose, statut tabagique, type de résidence, etc.

    Notre objectif est double : **prédire le risque d'AVC** et **détecter les biais potentiels**
    liés au **genre** (Male/Female) et à la **zone géographique** (Rural/Urban). Ces biais,
    s'ils existent dans les données d'entraînement, pourraient conduire un modèle à traiter
    inégalement différents groupes de population — avec des conséquences médicales réelles.
    """)

with col2:
    st.subheader("📊 Variables analysées")
    variables = {
        "Variable": ["age", "gender", "hypertension", "heart_disease", "avg_glucose_level",
                     "bmi", "smoking_status", "Residence_type", "stroke"],
        "Type": ["Numérique", "Catégorielle", "Binaire", "Binaire", "Numérique",
                 "Numérique", "Catégorielle", "Catégorielle", "🎯 Cible"],
        "Description": ["Âge du patient", "Male / Female", "0 ou 1", "0 ou 1",
                        "Glycémie moyenne", "Indice de masse corporelle",
                        "Statut tabagique", "Rural ou Urban", "AVC (1=oui, 0=non)"]
    }
    st.dataframe(pd.DataFrame(variables), width='stretch', hide_index=True)

st.markdown("---")

# ── Aperçu données ───────────────────────────────────────────────────────────
st.subheader("🔍 Aperçu des données")
st.dataframe(df.head(10), width='stretch')

st.info("👈 Utilisez la **barre latérale** pour naviguer entre les pages de l'application.")

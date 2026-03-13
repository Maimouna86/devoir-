"""
pages/1_Exploration.py — Page 2 : Exploration des données
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Exploration", page_icon="📊", layout="wide")

# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df = df[df['gender'] != 'Other'].copy()
    return df

df = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Exploration")
    st.markdown("---")
    st.subheader("🎛️ Filtres")

    gender_filter = st.multiselect(
        "Genre", options=df['gender'].unique(), default=list(df['gender'].unique())
    )
    residence_filter = st.multiselect(
        "Zone de résidence", options=df['Residence_type'].unique(),
        default=list(df['Residence_type'].unique())
    )
    age_range = st.slider("Tranche d'âge", int(df['age'].min()), int(df['age'].max()),
                          (0, int(df['age'].max())))
    stroke_filter = st.selectbox("Statut AVC", ["Tous", "AVC (1)", "Pas d'AVC (0)"])

df_f = df[
    df['gender'].isin(gender_filter) &
    df['Residence_type'].isin(residence_filter) &
    df['age'].between(age_range[0], age_range[1])
]
if stroke_filter == "AVC (1)":
    df_f = df_f[df_f['stroke'] == 1]
elif stroke_filter == "Pas d'AVC (0)":
    df_f = df_f[df_f['stroke'] == 0]

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📊 Exploration des Données")
st.caption(f"{len(df_f):,} patients affichés après filtres")
st.markdown("---")

# ── KPIs ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("👥 Patients filtrés", f"{len(df_f):,}")
c2.metric("📈 Âge moyen", f"{df_f['age'].mean():.1f} ans")
c3.metric("🩸 Glucose moyen", f"{df_f['avg_glucose_level'].mean():.1f} mg/dL")
c4.metric("⚖️ IMC médian", f"{df_f['bmi'].median():.1f}")

st.markdown("---")

# ── VIZ 1 : Distribution de la variable cible ────────────────────────────────
st.subheader("📌 Visualisation 1 — Distribution de la variable cible (AVC)")
col1, col2 = st.columns(2)

with col1:
    stroke_counts = df_f['stroke'].value_counts().reset_index()
    stroke_counts.columns = ['stroke', 'count']
    stroke_counts['label'] = stroke_counts['stroke'].map({0: "Pas d'AVC", 1: "AVC"})
    fig = px.pie(stroke_counts, values='count', names='label',
                 color_discrete_sequence=['#4CAF50', '#e94560'],
                 title="Répartition AVC vs Non-AVC")
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = px.histogram(df_f, x='age', color=df_f['stroke'].map({0: "Pas d'AVC", 1: "AVC"}),
                        barmode='overlay', nbins=30,
color_discrete_map={"Pas d'AVC": "#10b981", "AVC": "#ef4444"},
                        template="plotly_white",
                        title="Distribution d'âge selon le statut AVC",
                        labels={'color': 'Statut'})
    fig2.update_layout(xaxis_title="Âge", yaxis_title="Nombre de patients")
    st.plotly_chart(fig2, use_container_width=True)

# ── VIZ 2 : Comparaison par attribut sensible ────────────────────────────────
st.markdown("---")
st.subheader("📌 Visualisation 2 — Comparaison entre groupes (Genre & Résidence)")
col1, col2 = st.columns(2)

with col1:
    grp = df_f.groupby(['gender', 'stroke']).size().reset_index(name='count')
    grp['stroke_label'] = grp['stroke'].map({0: "Pas d'AVC", 1: "AVC"})
    fig = px.bar(grp, x='gender', y='count', color='stroke_label',
                 barmode='group',
                 color_discrete_map={"Pas d'AVC": "#4CAF50", "AVC": "#e94560"},
                 title="Cas d'AVC par Genre",
                 labels={'gender': 'Genre', 'count': 'Nombre', 'stroke_label': 'Statut'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    grp2 = df_f.groupby(['Residence_type', 'stroke']).size().reset_index(name='count')
    grp2['stroke_label'] = grp2['stroke'].map({0: "Pas d'AVC", 1: "AVC"})
    fig2 = px.bar(grp2, x='Residence_type', y='count', color='stroke_label',
                  barmode='group',
                  color_discrete_map={"Pas d'AVC": "#4CAF50", "AVC": "#e94560"},
                  title="Cas d'AVC par Zone de Résidence",
                  labels={'Residence_type': 'Zone', 'count': 'Nombre', 'stroke_label': 'Statut'})
    st.plotly_chart(fig2, use_container_width=True)

# ── VIZ 3 : Heatmap corrélations ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📌 Visualisation 3 — Matrice de corrélation (variables numériques)")

num_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke']
corr = df_f[num_cols].corr()
fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                title="Corrélations entre variables numériques")
fig.update_layout(height=450)
st.plotly_chart(fig, use_container_width=True)

# ── VIZ 4 : Scatter plot âge vs glucose ──────────────────────────────────────
st.markdown("---")
st.subheader("📌 Visualisation 4 — Relation Âge / Glycémie")

bmi_clean = df_f['bmi'].fillna(20).clip(10, 40)  # Clean BMI for size, reasonable range
fig = px.scatter(df_f, x='age', y='avg_glucose_level',
                 color=df_f['stroke'].map({0: "Pas d'AVC", 1: "AVC"}),
                 size=bmi_clean, hover_data=['gender', 'hypertension'],
                 color_discrete_map={"Pas d'AVC": "rgba(76,175,80,0.5)", "AVC": "#e94560"},
                 title="Âge vs Glycémie moyenne (taille = IMC)",
                 labels={'color': 'Statut', 'x': 'Âge', 'y': 'Glycémie moyenne (mg/dL)'})
fig.update_layout(height=450)
st.plotly_chart(fig, width='stretch')

# ── VIZ 5 : Box plots ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📌 Visualisation 5 — Distribution des variables cliniques par statut AVC")

var_choice = st.selectbox("Choisir une variable", ['age', 'avg_glucose_level', 'bmi'])
fig = px.box(df_f, x=df_f['stroke'].map({0: "Pas d'AVC", 1: "AVC"}), y=var_choice,
             color=df_f['stroke'].map({0: "Pas d'AVC", 1: "AVC"}),
             color_discrete_map={"Pas d'AVC": "#4CAF50", "AVC": "#e94560"},
             title=f"Distribution de {var_choice} selon le statut AVC",
             points='outliers')
fig.update_layout(xaxis_title="Statut AVC", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ── Statistiques descriptives ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Statistiques descriptives")
st.dataframe(df_f[num_cols].describe().round(2), width='stretch')

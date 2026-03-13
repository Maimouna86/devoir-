"""
pages/2_Detection_Biais.py — Page 3 : Détection de Biais
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.fairness import demographic_parity_difference, disparate_impact_ratio, equalized_odds

st.set_page_config(page_title="Détection de Biais", page_icon="⚠️", layout="wide")

# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df = df[df['gender'] != 'Other'].copy()
    return df

df = load_data()

# ── Header ───────────────────────────────────────────────────────────────────
st.title("⚠️ Détection de Biais")
st.markdown("""
Analyse des **biais potentiels** dans le dataset selon deux attributs sensibles :
**le genre** (Male/Female) et **la zone de résidence** (Rural/Urban).
""")
st.markdown("---")

# ── Sélection de l'attribut sensible ─────────────────────────────────────────
attr = st.radio(
    "🎯 Choisir l'attribut sensible à analyser :",
    ["Genre (gender)", "Zone géographique (Residence_type)"],
    horizontal=True
)

if attr == "Genre (gender)":
    sensitive_col = 'gender'
    unpriv = 'Female'
    priv   = 'Male'
    label  = "Genre"
else:
    sensitive_col = 'Residence_type'
    unpriv = 'Rural'
    priv   = 'Urban'
    label  = "Zone de résidence"

st.markdown("---")

# ── Explication du biais ──────────────────────────────────────────────────────
with st.expander("📖 Comprendre le biais analysé", expanded=True):
    if sensitive_col == 'gender':
        st.markdown("""
        **Attribut sensible : Genre (Male / Female)**

        Dans les datasets médicaux, le genre peut introduire un biais si les taux d'AVC
        observés diffèrent significativement entre hommes et femmes — non pas à cause de
        différences biologiques réelles, mais à cause de **sous-représentation**, de pratiques
        de collecte inégales, ou de facteurs sociaux confondants.

        Un modèle entraîné sur ces données biaisées pourrait **sous-estimer le risque d'AVC
        chez les femmes**, avec des conséquences médicales graves : diagnostic tardif,
        traitements insuffisants.

        **Métriques utilisées :**
        - *Parité Démographique* : les taux d'AVC doivent être proches entre groupes
        - *Impact Disproportionné* : ratio Female/Male des taux d'AVC
        """)
    else:
        st.markdown("""
        **Attribut sensible : Zone géographique (Rural / Urban)**

        Les patients ruraux et urbains peuvent avoir un accès différent aux soins de santé,
        ce qui influence à la fois la **prévention** et la **détection** des AVC. Si le
        dataset sur-représente les zones urbaines (souvent mieux équipées en hôpitaux),
        le modèle risque d'être moins précis pour les patients ruraux.

        Un tel biais perpétuerait des **inégalités de santé territoriales** : les populations
        rurales, déjà défavorisées dans l'accès aux soins, recevraient des prédictions
        moins fiables.

        **Métriques utilisées :**
        - *Parité Démographique* : comparaison des taux d'AVC Rural vs Urban
        - *Impact Disproportionné* : ratio Rural/Urban
        """)

# ── Taux d'AVC par groupe ─────────────────────────────────────────────────────
st.subheader(f"📊 Taux d'AVC par {label}")

group_stats = df.groupby(sensitive_col).agg(
    Total=('stroke', 'count'),
    AVC=('stroke', 'sum'),
).reset_index()
group_stats['Taux_AVC (%)'] = (group_stats['AVC'] / group_stats['Total'] * 100).round(2)

col1, col2 = st.columns(2)
with col1:
    st.dataframe(group_stats, width='stretch', hide_index=True)

with col2:
    fig = px.bar(group_stats, x=sensitive_col, y='Taux_AVC (%)',
                 color=sensitive_col,
                 color_discrete_sequence=['#ef4444', '#3b82f6'],
                 template='plotly_white',
                 text='Taux_AVC (%)',
                 title=f"Taux d'AVC (%) par {label}")
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(showlegend=False, yaxis=dict(range=[0, group_stats['Taux_AVC (%)'].max() * 1.4]))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Métriques de Fairness ─────────────────────────────────────────────────────
st.subheader("📐 Métriques de Fairness")

y_true = df['stroke'].values
y_pred = df['stroke'].values   # On analyse les données brutes (pas un modèle)
sensitive = df[sensitive_col].values

# Métrique 1 : Parité Démographique
dp = demographic_parity_difference(y_true, y_pred, sensitive)
# Métrique 2 : Impact Disproportionné
di = disparate_impact_ratio(y_true, y_pred, sensitive,
                             unprivileged_value=unpriv,
                             privileged_value=priv)

m1, m2, m3 = st.columns(3)

diff_val = dp['difference'] * 100
m1.metric(
    "📏 Différence de Parité Démographique",
    f"{diff_val:.2f}%",
    help="Différence absolue des taux d'AVC entre groupes. Idéal = 0%."
)

ratio_val = di['ratio']
delta_color = "normal" if 0.8 <= ratio_val <= 1.2 else "inverse"
m2.metric(
    "⚖️ Ratio d'Impact Disproportionné",
    f"{ratio_val:.3f}",
    delta="✅ OK (>0.8)" if ratio_val >= 0.8 else "⚠️ Discriminatoire (<0.8)",
    delta_color=delta_color,
    help="Ratio taux groupe non-privilégié / groupe privilégié. Idéal = 1.0. < 0.8 = discriminatoire."
)

m3.metric(
    f"🔵 Taux AVC — {unpriv}",
    f"{di['rate_unprivileged']*100:.2f}%",
    delta=f"vs {priv} : {di['rate_privileged']*100:.2f}%",
    delta_color="off"
)

# Interprétation automatique
st.markdown("---")
st.subheader("🔍 Interprétation")

if ratio_val < 0.8:
    bias_level = "🔴 **Biais significatif détecté**"
    bias_color = "#ffcccc"
elif ratio_val < 0.9:
    bias_level = "🟠 **Biais modéré détecté**"
    bias_color = "#fff3cd"
else:
    bias_level = "🟢 **Pas de biais significatif**"
    bias_color = "#d4edda"

st.markdown(f"""
<div style="background:{bias_color}; padding:1.2rem; border-radius:8px; margin-bottom:1rem;">
{bias_level}
</div>
""", unsafe_allow_html=True)

if sensitive_col == 'gender':
    st.markdown(f"""
    Le taux d'AVC observé est de **{di['rate_privileged']*100:.2f}%** chez les hommes et
    **{di['rate_unprivileged']*100:.2f}%** chez les femmes, soit une différence de
    **{diff_val:.2f} points de pourcentage**.

    Le ratio d'impact disproportionné est de **{ratio_val:.3f}** (Female/Male).
    {'Un ratio inférieur à 0.8 indique une discrimination systémique selon la règle des 4/5ème.' if ratio_val < 0.8 else 'Le ratio est supérieur à 0.8, ce qui est dans la zone acceptable selon la règle des 4/5ème.'}

    **Groupe potentiellement défavorisé** : {'les **femmes** (taux d\'AVC plus faible dans le dataset, risque de sous-diagnostic)' if di['rate_unprivileged'] < di['rate_privileged'] else 'les **hommes** (taux d\'AVC plus faible dans le dataset)'}.

    **Impact réel** : Un modèle entraîné sur ces données sans correction pourrait moins bien
    détecter les AVC chez le groupe sous-représenté, retardant le diagnostic et la prise en charge.
    **Recommandation** : Rééchantillonnage (SMOTE), pondération des classes, ou algorithmes
    de fairness (Reweighing, Adversarial Debiasing).
    """)
else:
    st.markdown(f"""
    Le taux d'AVC observé est de **{di['rate_privileged']*100:.2f}%** en zone urbaine et
    **{di['rate_unprivileged']*100:.2f}%** en zone rurale, soit une différence de
    **{diff_val:.2f} points de pourcentage**.

    Le ratio d'impact disproportionné Rural/Urban est de **{ratio_val:.3f}**.
    {'Un ratio inférieur à 0.8 indique une disparité préoccupante.' if ratio_val < 0.8 else 'Le ratio est proche de 1, les deux zones sont équitablement représentées.'}

    **Groupe potentiellement défavorisé** : {'les patients **ruraux** (moins bien représentés)' if di['rate_unprivileged'] < di['rate_privileged'] else 'les patients **urbains**'}.

    **Impact réel** : Les inégalités d'accès aux soins en zones rurales peuvent biaiser
    le dataset — les cas moins diagnostiqués n'apparaissent pas dans les données.
    **Recommandation** : Stratification géographique lors de la collecte, validation croisée
    par zone, et audit régulier du modèle sur les sous-groupes.
    """)

st.markdown("---")

# ── Visualisation détaillée par groupe ───────────────────────────────────────
st.subheader(f"📊 Analyse détaillée : facteurs de risque par {label}")

risk_var = st.selectbox("Variable de risque à comparer", ['age', 'avg_glucose_level', 'bmi', 'hypertension'])

col1, col2 = st.columns(2)
with col1:
    fig = px.box(df[df['stroke'] == 1], x=sensitive_col, y=risk_var,
                 color=sensitive_col,
                 title=f"{risk_var} — Patients avec AVC uniquement",
                 color_discrete_sequence=['#ef4444', '#3b82f6'],
                        template='plotly_white')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Taux d'AVC par groupe × facteur de risque catégoriel
    if risk_var == 'hypertension':
        cross = df.groupby([sensitive_col, 'hypertension'])['stroke'].mean().reset_index()
        cross['hypertension'] = cross['hypertension'].map({0: 'Non', 1: 'Oui'})
        cross['Taux AVC (%)'] = cross['stroke'] * 100
        fig2 = px.bar(cross, x=sensitive_col, y='Taux AVC (%)', color='hypertension',
                      barmode='group', title="Taux d'AVC : Hypertension × Groupe",
                      color_discrete_sequence=['#10b981', '#ef4444'],
                        template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        fig2 = px.violin(df, x=sensitive_col, y=risk_var,
                         color=df['stroke'].map({0: "Pas d'AVC", 1: "AVC"}),
                         box=True,
                         color_discrete_map={"Pas d'AVC": "#10b981", "AVC": "#ef4444"},
                         template='plotly_white',
                         title=f"Distribution de {risk_var} par {label} et statut AVC")
        st.plotly_chart(fig2, use_container_width=True)

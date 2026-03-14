# Acceuil.py (adaptatif clair/sombre)
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Bias Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dataset ──────────────────────────────────────────────────────────────────
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
        f"📁 Fichier introuvable : `{DATA_PATH.name}` introuvable. "
        "Ajoute-le au projet ou adapte DATA_PATH."
    )
    st.stop()

# ── Helpers thème adaptatif ──────────────────────────────────────────────────
def is_dark_theme() -> bool:
    base = st.get_option("theme.base")
    # base peut être None si pas de config; on considère clair par défaut
    return str(base).lower() == "dark"

def get_palette():
    if is_dark_theme():
        return {
            "text": "#e5e7eb",
            "muted": "#cbd5e1",
            "card_bg": "#0f172a",
            "border": "rgba(148,163,184,0.18)",
            "kpi_title": "#cbd5e1",
            "kpi_value": "#e5e7eb",
        }
    else:
        return {
            "text": "#0f172a",
            "muted": "#475569",
            "card_bg": "#ffffff",
            "border": "#e2e8f0",
            "kpi_title": "#475569",
            "kpi_value": "#0f172a",
        }

PALETTE = get_palette()

def set_plotly_template(fig, height=None):
    template = "plotly_dark" if is_dark_theme() else "plotly_white"
    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if height:
        fig.update_layout(height=height)
    return fig

def style_df(df_in: pd.DataFrame):
    """Styler pandas qui suit le thème courant (clair/sombre)."""
    header_bg = "#111827" if is_dark_theme() else "#f8fafc"
    return (
        df_in.style
        .set_table_styles([
            {"selector": "th", "props": [("background", header_bg),
                                         ("color", PALETTE["text"]),
                                         ("border", f"1px solid {PALETTE['border']}")]},
            {"selector": "td", "props": [("background", PALETTE["card_bg"]),
                                         ("color", PALETTE["text"]),
                                         ("border", f"1px solid {PALETTE['border']}")]},
        ])
        .set_properties(**{
            "background-color": PALETTE["card_bg"],
            "color": PALETTE["text"],
            "border": f"1px solid {PALETTE['border']}",
        })
    )

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Healthcare Bias Analysis")
    st.markdown("---")
    st.caption("Dataset : Stroke Prediction | Kaggle")
    st.caption("Parcours A — Mastère 2 Data & IA")

# ── En-tête (sans forcer de fond) ────────────────────────────────────────────
st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 50%, #1e40af 100%);
        color: white; text-align: center; border-radius: 24px;
        padding: 1.5rem 1rem; margin-bottom: 1.5rem;
        box-shadow: 0 6px 25px rgba(0,0,0,0.15);
    ">
      <h1 style="font-size:3rem; margin:0;">Healthcare Bias Analysis</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ── KPIs ─────────────────────────────────────────────────────────────────────
total = len(df)
n_stroke = int(df["stroke"].sum())
stroke_rate = n_stroke / total * 100 if total else 0.0
missing_bmi = int(df["bmi"].isnull().sum())
missing_pct = missing_bmi / total * 100 if total else 0.0

c1, c2, c3, c4 = st.columns(4)
kpi_css = f"""
<style>
/* Légère carte autour des metrics, couleurs suivent le thème */
[data-testid="stMetric"] {{
  background: {PALETTE["card_bg"]};
  padding: 1rem; border-radius: 16px;
  border: 1px solid {PALETTE["border"]};
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}}
[data-testid="stMetric"] > label {{ color: {PALETTE["kpi_title"]} !important; }}
[data-testid="stMetric"] > div > div {{ color: {PALETTE["kpi_value"]} !important; }}
</style>
"""
st.markdown(kpi_css, unsafe_allow_html=True)

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
- Identifier l’influence du **genre** et de la **résidence** (Rural/Urban)

L’objectif : améliorer l’équité et la fiabilité des prédictions.
""")

st.markdown("---")

# ── Aperçu des données ──────────────────────────────────────────────────────
st.subheader("Aperçu des données")
st.dataframe(style_df(df.head(10)), width='stretch')
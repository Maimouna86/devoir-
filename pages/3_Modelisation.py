import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from utils.fairness import demographic_parity_difference, disparate_impact_ratio, equalized_odds

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import numpy as np

st.set_page_config(page_title="Modélisation", page_icon="🤖", layout="wide")

# ── Data & preprocessing ─────────────────────────────────────────────────────
@st.cache_data
def load_and_prep():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df = df[df['gender'] != 'Other'].copy()
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

    # Encodage
    df['gender_enc']    = (df['gender'] == 'Male').astype(int)
    df['married_enc']   = (df['ever_married'] == 'Yes').astype(int)
    df['residence_enc'] = (df['Residence_type'] == 'Urban').astype(int)

    work_dummies  = pd.get_dummies(df['work_type'], prefix='work', drop_first=True)
    smoke_dummies = pd.get_dummies(df['smoking_status'], prefix='smoke', drop_first=True)
    df = pd.concat([df, work_dummies, smoke_dummies], axis=1)
    return df

df = load_and_prep()

feature_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
                'gender_enc', 'married_enc', 'residence_enc'] + \
               [c for c in df.columns if c.startswith('work_') or c.startswith('smoke_')]
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df['stroke']

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🤖 Modélisation")
st.markdown("Entraînement d'un modèle de classification et analyse des biais sur les **prédictions**.")
st.markdown("---")

# ── Paramètres ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Paramètres")
    model_choice = st.selectbox("Modèle", ["Logistic Regression", "Random Forest"])
    use_smote    = st.checkbox("Rééchantillonnage SMOTE (déséquilibre)", value=True)
    test_size    = st.slider("Taille du jeu de test (%)", 10, 40, 20)
    sensitive_attr = st.radio("Attribut sensible", ["gender", "Residence_type"])

# ── Entraînement ─────────────────────────────────────────────────────────────
@st.cache_data
def train_model(model_name, smote, ts):
    X_numeric = X.select_dtypes(include=[np.number])  # Only numeric cols
    feature_cols_used = X_numeric.columns.tolist()  # Update for length match
    X_tr, X_te, y_tr, y_te = train_test_split(X_numeric, y, test_size=ts/100,
                                               random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    if smote:
        sm = SMOTE(random_state=42)
        X_tr_s, y_tr = sm.fit_resample(X_tr_s, y_tr)

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    y_prob = model.predict_proba(X_te_s)[:, 1]

    # Récupérer l'index du test set
    _, test_idx = train_test_split(df.index, test_size=ts/100, random_state=42,
                                   stratify=y)
    return y_te, y_pred, y_prob, model, test_idx, feature_cols_used

y_te, y_pred, y_prob, model, test_idx, feature_cols_used = train_model(model_choice, use_smote, test_size)
df_test = df.loc[test_idx].copy()
df_test['y_pred'] = y_pred

# ── Métriques globales ───────────────────────────────────────────────────────
st.subheader("📊 Performances globales du modèle")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy",  f"{accuracy_score(y_te, y_pred):.3f}")
c2.metric("Precision", f"{precision_score(y_te, y_pred, zero_division=0):.3f}")
c3.metric("Recall",    f"{recall_score(y_te, y_pred, zero_division=0):.3f}")
c4.metric("F1-Score",  f"{f1_score(y_te, y_pred, zero_division=0):.3f}")
c5.metric("ROC-AUC",   f"{roc_auc_score(y_te, y_prob):.3f}")

st.markdown("---")

# ── Confusion matrix globale ─────────────────────────────────────────────────
st.subheader("🔲 Matrice de confusion globale")
cm = confusion_matrix(y_te, y_pred)
fig = px.imshow(cm, text_auto=True,
                x=["Prédit: Pas d'AVC", "Prédit: AVC"],
                y=["Réel: Pas d'AVC", "Réel: AVC"],
                color_continuous_scale='Blues',
                title="Matrice de confusion — ensemble de test")
fig.update_layout(height=350)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# ── Fairness sur prédictions ──────────────────────────────────────────────────
st.subheader(f"⚖️ Métriques de Fairness sur les prédictions — {sensitive_attr}")

sensitive_test = df_test[sensitive_attr].values
y_te_arr  = np.array(y_te)
y_pred_arr = np.array(y_pred)

groups = df_test[sensitive_attr].unique()
unpriv = 'Female' if sensitive_attr == 'gender' else 'Rural'
priv   = 'Male'   if sensitive_attr == 'gender' else 'Urban'

dp = demographic_parity_difference(y_te_arr, y_pred_arr, sensitive_test)
di = disparate_impact_ratio(y_te_arr, y_pred_arr, sensitive_test,
                             unprivileged_value=unpriv, privileged_value=priv)
eo = equalized_odds(y_te_arr, y_pred_arr, sensitive_test)

m1, m2, m3 = st.columns(3)
m1.metric("Différence de Parité", f"{dp['difference']*100:.2f}%",
          help="Différence des taux de prédiction positive entre groupes.")
m2.metric("Ratio Impact Disproportionné", f"{di['ratio']:.3f}",
          delta="✅ OK" if di['ratio'] >= 0.8 else "⚠️ < 0.8",
          delta_color="normal" if di['ratio'] >= 0.8 else "inverse")
m3.metric(f"Taux prédit AVC — {unpriv}", f"{di['rate_unprivileged']*100:.2f}%",
          delta=f"vs {priv} : {di['rate_privileged']*100:.2f}%", delta_color="off")

# ── Confusion matrices par groupe ────────────────────────────────────────────
st.markdown("---")
st.subheader(f"🔲 Matrices de confusion par groupe ({sensitive_attr})")

cols = st.columns(len(groups))
for i, group in enumerate(sorted(groups)):
    mask = df_test[sensitive_attr] == group
    yt_g  = y_te_arr[mask]
    yp_g  = y_pred_arr[mask]
    cm_g  = confusion_matrix(yt_g, yp_g, labels=[0, 1])
    with cols[i]:
        fig = px.imshow(cm_g, text_auto=True,
                        x=["Prédit: 0", "Prédit: 1"],
                        y=["Réel: 0", "Réel: 1"],
                        color_continuous_scale='Oranges',
                        title=f"Groupe : {group} (n={mask.sum()})")
        fig.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

        acc_g  = accuracy_score(yt_g, yp_g)
        rec_g  = recall_score(yt_g, yp_g, zero_division=0)
        prec_g = precision_score(yt_g, yp_g, zero_division=0)
        st.caption(f"Accuracy: {acc_g:.3f} | Recall: {rec_g:.3f} | Precision: {prec_g:.3f}")

# ── Equalized Odds ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📐 Equalized Odds (TPR / FPR par groupe)")

eo_df = pd.DataFrame(eo).T.reset_index()
eo_df.columns = ['Groupe', 'TPR (Recall)', 'FPR']
st.dataframe(eo_df, width='stretch', hide_index=True)

col1, col2 = st.columns(2)
with col1:
    fig = px.bar(eo_df, x='Groupe', y='TPR (Recall)', color='Groupe',
                 title="True Positive Rate par groupe",
                 color_discrete_sequence=['#e94560', '#4C72B0'],
                 text='TPR (Recall)')
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1.2]))
    st.plotly_chart(fig, width='stretch')

with col2:
    fig2 = px.bar(eo_df, x='Groupe', y='FPR', color='Groupe',
                  title="False Positive Rate par groupe",
                  color_discrete_sequence=['#e94560', '#4C72B0'],
                  text='FPR')
    fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig2.update_layout(showlegend=False, yaxis=dict(range=[0, max(eo_df['FPR'].max() * 1.4, 0.1)]))
    st.plotly_chart(fig2, width='stretch')

# ── Feature importance (RF) ──────────────────────────────────────────────────
if model_choice == "Random Forest":
    st.markdown("---")
    st.subheader("🌳 Importance des variables (Random Forest)")
    if hasattr(model, 'feature_importances_') and len(feature_cols_used) == len(model.feature_importances_):
        fi = pd.DataFrame({'Feature': feature_cols_used, 'Importance': model.feature_importances_})
        fi = fi.sort_values('Importance', ascending=True).tail(12)
        fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Reds',
                     title="Top 12 variables les plus importantes")
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Feature importance disponible seulement pour Random Forest.")

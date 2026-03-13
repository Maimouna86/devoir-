# 🧠 Stroke Bias Detector

Application Streamlit pour l'analyse et la détection de biais dans la prédiction du risque d'AVC.

**Mastère 2 — Data & Intelligence Artificielle | Parcours A**

---

## 🚀 Application en ligne

👉 **[Accéder à l'application](https://your-app-url.streamlit.app)** *(à compléter après déploiement)*

👉 **[Repo GitHub](https://github.com/loic00l/Healthcare)**

---

## 📁 Structure du projet

```
stroke_app/
├── app.py                          # 🏠 Page Accueil
├── pages/
│   ├── 1_Exploration.py            # 📊 Exploration des données
│   ├── 2_Detection_Biais.py        # ⚠️  Détection de Biais
│   └── 3_Modelisation.py           # 🤖 Modélisation & Fairness
├── utils/
│   ├── fairness.py                 # Métriques de fairness
│   └── data_loader.py              # Chargement & preprocessing
├── healthcare-dataset-stroke-data.csv
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source** : [Stroke Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Lignes** : 5 109 patients
- **Variables** : 12 (âge, genre, hypertension, maladie cardiaque, glucose, IMC, tabagisme...)
- **Cible** : `stroke` (1 = AVC, 0 = Pas d'AVC)

---

## 📄 Pages de l'application

| Page | Contenu |
|------|---------|
| 🏠 Accueil | Présentation, KPIs, aperçu données |
| 📊 Exploration | 5 visualisations interactives, filtres genre/résidence/âge |
| ⚠️ Détection de Biais | Parité démographique, impact disproportionné, interprétation |
| 🤖 Modélisation | LR / RF, métriques de fairness, confusion matrices par groupe |

---

## ⚙️ Installation locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔍 Biais analysés

- **Genre** (Male / Female) : différences dans les taux d'AVC et les prédictions du modèle
- **Zone géographique** (Rural / Urban) : disparités d'accès aux soins et impact sur les prédictions

---

## 👤 Auteur

Nom : *(à compléter)*  
Partenaire : *(à compléter si binôme)*

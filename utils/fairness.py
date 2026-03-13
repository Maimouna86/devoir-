"""
utils/fairness.py
-----------------
Métriques de fairness pour la détection de biais.
"""

import numpy as np
import pandas as pd


def demographic_parity_difference(y_true, y_pred, sensitive_attribute):
    """
    Calcule la différence de parité démographique entre groupes.
    Résultat idéal = 0 (pas de biais).
    """
    groups = np.unique(sensitive_attribute)
    rates = {}
    for g in groups:
        mask = sensitive_attribute == g
        if mask.sum() == 0:
            continue
        rates[g] = y_pred[mask].mean()

    if len(rates) < 2:
        return {"rates": rates, "difference": 0.0, "groups": list(rates.keys())}

    vals = list(rates.values())
    diff = max(vals) - min(vals)
    return {"rates": rates, "difference": float(diff), "groups": list(rates.keys())}


def disparate_impact_ratio(y_true, y_pred, sensitive_attribute,
                           unprivileged_value, privileged_value):
    """
    Calcule le ratio d'impact disproportionné.
    Résultat idéal = 1.0 ; < 0.8 considéré comme discriminatoire.
    """
    mask_unpriv = sensitive_attribute == unprivileged_value
    mask_priv   = sensitive_attribute == privileged_value

    rate_unpriv = y_pred[mask_unpriv].mean() if mask_unpriv.sum() > 0 else 0
    rate_priv   = y_pred[mask_priv].mean()   if mask_priv.sum()   > 0 else 1

    ratio = rate_unpriv / rate_priv if rate_priv > 0 else 0
    return {
        "rate_unprivileged": float(rate_unpriv),
        "rate_privileged":   float(rate_priv),
        "ratio":             float(ratio),
    }


def equalized_odds(y_true, y_pred, sensitive_attribute):
    """
    Calcule le True Positive Rate et False Positive Rate par groupe.
    """
    groups = np.unique(sensitive_attribute)
    results = {}
    for g in groups:
        mask = sensitive_attribute == g
        yt, yp = y_true[mask], y_pred[mask]
        tp = ((yt == 1) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        tn = ((yt == 0) & (yp == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        results[g] = {"TPR": round(tpr, 4), "FPR": round(fpr, 4)}
    return results

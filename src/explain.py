"""Explicabilité : valeurs SHAP exactes pour XGBoost (pred_contribs) + codes motifs."""
import numpy as np
import pandas as pd
import xgboost as xgb
from .features import LABELS


def shap_values(model, X: pd.DataFrame):
    """Retourne (valeurs SHAP [n, p], base_value). Marge en log-odds.
    pred_contribs = TreeSHAP natif de XGBoost, identique à shap.TreeExplainer, sans dépendance de version."""
    booster = model.get_booster()
    contrib = booster.predict(xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)
    return contrib[:, :-1], float(contrib[0, -1])


def _fmt(feature, value):
    if feature.startswith("cat_") or feature in ("is_weekend", "is_night", "gender_m"):
        return "oui" if value >= 0.5 else "non"
    if feature == "log_amt":
        return f"{np.expm1(value):,.2f} $"
    if feature == "log_gap_h":
        return f"{np.expm1(value):.1f} h"
    if feature == "dist_km":
        return f"{value:.0f} km"
    if feature == "log_city_pop":
        return f"{np.expm1(value):,.0f} hab."
    if feature in ("hour_sin", "hour_cos"):
        return f"{value:.2f}"
    return f"{value:.2f}"


def reason_codes(shap_row, x_row: pd.Series, top=4):
    """Les 3 à 5 variables qui ont le plus POUSSÉ le score vers « fraude » (SHAP > 0)."""
    order = np.argsort(-shap_row)[:top]
    codes, seen = [], set()
    for i in order:
        if shap_row[i] <= 0:
            break
        f = x_row.index[i]
        label = LABELS.get(f, f)
        if label in seen:      # hour_sin / hour_cos -> un seul motif « Heure »
            continue
        seen.add(label)
        codes.append({"feature": f, "motif": label, "valeur": _fmt(f, x_row.iloc[i]),
                      "contribution_shap": round(float(shap_row[i]), 3)})
    return codes

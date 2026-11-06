"""Module 6 — API FastAPI : score une transaction, renvoie alerte + codes motifs + SHAP.

Lancer :  uvicorn api.main:app --reload      puis  http://127.0.0.1:8000/docs
Prérequis : python run_pipeline.py (génère artifacts/).
"""
import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config as C
from src.explain import reason_codes, shap_values
from src.features import FEATURES, LABELS, add_stateless_features

app = FastAPI(title="API de détection de fraude explicable", version="1.0",
              description="Challenge 2 – Data Mining ENSI. Modèle XGBoost + explications SHAP.")
_state = {}


def _load():
    if not _state:
        try:
            _state["model"] = joblib.load(C.ART_DIR / "xgb_final.joblib")
            _state["res"] = json.load(open(C.ART_DIR / "results.json"))
        except FileNotFoundError:
            raise HTTPException(503, "Artefacts absents : lancez d'abord `python run_pipeline.py`.")
    return _state


class Transaction(BaseModel):
    trans_date_trans_time: str = Field(..., examples=["2020-06-21 02:14:00"])
    amt: float = Field(..., gt=0, examples=[842.5])
    category: str = Field(..., examples=["shopping_net"])
    gender: str = "F"
    lat: float; long: float; merch_lat: float; merch_long: float
    city_pop: int = 10000
    dob: str = Field(..., examples=["1985-04-12"])
    # Variables comportementales : fournies par le système appelant (feature store).
    # Si absentes, on utilise la médiane d'entraînement (profil « neutre »).
    n_tx_1h: Optional[float] = None
    n_tx_24h: Optional[float] = None
    amt_ratio_30d: Optional[float] = None
    time_since_prev_h: Optional[float] = None


def _to_features(tx: Transaction, res) -> pd.DataFrame:
    d = tx.model_dump()
    df = add_stateless_features(pd.DataFrame([d]))
    dflt = res["meta"]["behavioral_defaults"]
    df["n_tx_1h"] = tx.n_tx_1h if tx.n_tx_1h is not None else dflt["n_tx_1h"]
    df["n_tx_24h"] = tx.n_tx_24h if tx.n_tx_24h is not None else dflt["n_tx_24h"]
    df["amt_ratio_30d"] = tx.amt_ratio_30d if tx.amt_ratio_30d is not None else dflt["amt_ratio_30d"]
    df["log_gap_h"] = np.log1p(min(tx.time_since_prev_h, 720.0)) if tx.time_since_prev_h is not None else dflt["log_gap_h"]
    return df[FEATURES].astype("float32")


class Prediction(BaseModel):
    fraud_probability: float
    alert: bool
    threshold: float
    reason_codes: list = []


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": (C.ART_DIR / "xgb_final.joblib").exists()}


@app.get("/model-info")
def model_info():
    r = _load()["res"]
    x = r["models"]["XGBoost"]
    return {"threshold": x["threshold"], "review_cost": r["meta"]["review_cost"],
            "periods": r["meta"]["period"], "imbalance_strategy": r["meta"]["imbalance_best"],
            "test_metrics": x["test"], "features": FEATURES}


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction, threshold: Optional[float] = None):
    """Score + décision d'alerte (seuil par défaut = seuil optimal en coût, choisi sur la validation)."""
    s = _load(); X = _to_features(tx, s["res"])
    p = float(s["model"].predict_proba(X)[0, 1])
    thr = threshold if threshold is not None else (s["res"]["models"]["XGBoost"]["threshold"] or 1.0)
    codes = []
    if p >= thr:
        sh, _ = shap_values(s["model"], X)
        codes = reason_codes(sh[0], X.iloc[0], top=4)
    return Prediction(fraud_probability=p, alert=p >= thr, threshold=thr, reason_codes=codes)


@app.post("/explain")
def explain(tx: Transaction):
    """Contributions SHAP complètes (log-odds) pour une transaction."""
    s = _load(); X = _to_features(tx, s["res"])
    sh, base = shap_values(s["model"], X)
    contrib = sorted(({"feature": f, "label": LABELS.get(f, f), "value": float(X.iloc[0][f]),
                       "shap": float(v)} for f, v in zip(FEATURES, sh[0])), key=lambda d: -abs(d["shap"]))
    return {"base_value_logodds": base, "fraud_probability": float(s["model"].predict_proba(X)[0, 1]),
            "contributions": contrib, "reason_codes": reason_codes(sh[0], X.iloc[0], top=4)}


@app.post("/predict/batch")
def predict_batch(txs: list[Transaction]):
    s = _load(); thr = s["res"]["models"]["XGBoost"]["threshold"] or 1.0
    X = pd.concat([_to_features(t, s["res"]) for t in txs], ignore_index=True)
    p = s["model"].predict_proba(X)[:, 1]
    return [{"fraud_probability": float(v), "alert": bool(v >= thr)} for v in p]

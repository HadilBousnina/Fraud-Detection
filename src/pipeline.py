"""Pipeline complet : données -> variables -> split chrono -> déséquilibre -> modèles -> seuil -> artefacts."""
import json, time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config as C
from .data import chronological_split, load_raw
from .explain import shap_values
from .features import FEATURES, build_features
from .metrics import best_threshold, cost_curve, evaluate, precision_recall_at_k

MODELS = {  # nom affiché -> colonne de score
    "Régression logistique pondérée": "score_lr",
    "XGBoost": "score_xgb",
    "Isolation Forest": "score_if",
    "LOF (novelty)": "score_lof",
    "Autoencodeur (MLP)": "score_ae",
}
TYPES = {"score_lr": "Supervisé", "score_xgb": "Supervisé", "score_if": "Non supervisé",
         "score_lof": "Non supervisé", "score_ae": "Non supervisé"}


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_eda(df):
    y = df[C.TARGET].to_numpy(); t = pd.to_datetime(df[C.TIME_COL])
    edges = np.linspace(0, np.log10(df["amt"].max() + 1), 31)
    h0, _ = np.histogram(np.log10(df.loc[y == 0, "amt"] + 1), bins=edges, density=True)
    h1, _ = np.histogram(np.log10(df.loc[y == 1, "amt"] + 1), bins=edges, density=True)

    d = df.assign(_m=t.dt.to_period("M").astype(str), _h=t.dt.hour, _dow=t.dt.dayofweek)
    return {
        "n": int(len(df)), "n_fraud": int(y.sum()), "fraud_rate": float(y.mean()),
        "date_min": str(t.min()), "date_max": str(t.max()),
        "amt_mean": {"legit": float(df.loc[y == 0, "amt"].mean()), "fraud": float(df.loc[y == 1, "amt"].mean())},
        "amt_median": {"legit": float(df.loc[y == 0, "amt"].median()), "fraud": float(df.loc[y == 1, "amt"].median())},
        "amt_hist": {"edges": edges.tolist(), "legit": h0.tolist(), "fraud": h1.tolist()},
        "by_month": rate_of(d, "_m"),
        "by_hour": rate_of(d, "_h"), "by_category": rate_of(d, "category"), "by_dow": rate_of(d, "_dow"),
    }


def rate_of(d, key):
    g = d.groupby(key)[C.TARGET].agg(["mean", "sum", "count"])
    return {"index": [str(i) for i in g.index], "rate": g["mean"].tolist(),
            "frauds": g["sum"].tolist(), "count": g["count"].tolist()}


def xgb_model(spw=1.0):
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
                         eval_metric="aucpr", n_jobs=-1, random_state=C.RANDOM_STATE)


def compare_imbalance(Xtr, ytr, Xva, yva, k_val):
    """Stratégies appliquées au SEUL jeu d'entraînement, comparées en validation."""
    rng = np.random.RandomState(C.RANDOM_STATE)
    pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    spw = len(neg) / max(len(pos), 1)
    keep = np.concatenate([pos, rng.choice(neg, size=min(len(neg), 10 * len(pos)), replace=False)])
    strategies = {
        "Aucune": (Xtr, ytr, 1.0),
        "Pondération (scale_pos_weight)": (Xtr, ytr, spw),
        "Sous-échantillonnage 10:1": (Xtr.iloc[keep], ytr[keep], 1.0),
    }
    rows, models = [], {}
    from sklearn.metrics import average_precision_score
    for name, (X, y, w) in strategies.items():
        t0 = time.time(); m = xgb_model(w).fit(X, y)
        s = m.predict_proba(Xva)[:, 1]
        p, r = precision_recall_at_k(yva, s, k_val)
        rows.append({"strategie": name, "n_train": int(len(y)), "pr_auc_val": float(average_precision_score(yva, s)),
                     "precision_at_k_val": float(p), "recall_at_k_val": float(r), "temps_s": round(time.time() - t0, 1)})
        models[name] = m
        _log(f"  déséquilibre « {name} » : PR-AUC val = {rows[-1]['pr_auc_val']:.4f}")
    best = max(rows, key=lambda r: r["pr_auc_val"])["strategie"]
    return rows, best, models[best], strategies[best]


def run(data=None, max_rows=None):
    C.ART_DIR.mkdir(exist_ok=True); C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    _log("Chargement des données...")
    df = load_raw(data, max_rows)
    eda = make_eda(df)
    _log(f"{len(df):,} transactions, taux de fraude = {eda['fraud_rate']:.3%}")
    _log("Construction des variables (temps, géographie, vélocité)...")
    df = build_features(df)
    tr, va, te = chronological_split(df, C.SPLIT)
    for n, d in (("train", tr), ("val", va), ("test", te)):
        _log(f"  {n}: {len(d):,} lignes | {d[C.TIME_COL].min():%Y-%m-%d} -> {d[C.TIME_COL].max():%Y-%m-%d} | fraude {d[C.TARGET].mean():.3%}")
    Xtr, Xva, Xte = (d[FEATURES].astype("float32") for d in (tr, va, te))
    ytr, yva, yte = (d[C.TARGET].to_numpy() for d in (tr, va, te))
    k_val, k_te = int(C.BUDGET_FRAC * len(va)), int(C.BUDGET_FRAC * len(te))

    _log("Comparaison des stratégies contre le déséquilibre (validation)...")
    imb_rows, imb_best, xgb_final, (Xb, yb, _) = compare_imbalance(Xtr, ytr, Xva, yva, k_val)
    _log(f"  -> stratégie retenue : {imb_best}")

    sv, st = {}, {}
    # 1) XGBoost
    sv["score_xgb"], st["score_xgb"] = (xgb_final.predict_proba(X)[:, 1] for X in (Xva, Xte))
    # 2) Régression logistique pondérée
    _log("Régression logistique pondérée...")
    scaler = StandardScaler().fit(Xtr)
    z = lambda X: np.clip(scaler.transform(X), -10, 10)
    lr = LogisticRegression(class_weight="balanced", max_iter=500).fit(z(Xtr), ytr)
    sv["score_lr"], st["score_lr"] = (lr.predict_proba(z(X))[:, 1] for X in (Xva, Xte))
    # 3) Isolation Forest (sans étiquettes)
    _log("Isolation Forest...")
    iso = IsolationForest(n_estimators=200, max_samples=min(10_000, len(Xtr)), random_state=C.RANDOM_STATE, n_jobs=-1).fit(z(Xtr))
    sv["score_if"], st["score_if"] = (-iso.score_samples(z(X)) for X in (Xva, Xte))
    # 4) LOF novelty (sous-échantillon : coûteux)
    _log("LOF...")
    rng = np.random.RandomState(C.RANDOM_STATE)
    sub = rng.choice(len(Xtr), size=min(C.LOF_FIT_SIZE, len(Xtr)), replace=False)
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1).fit(z(Xtr.iloc[sub]))
    def lof_score(X):
        Z = z(X); return np.concatenate([-lof.score_samples(Z[i:i + 50_000]) for i in range(0, len(Z), 50_000)])
    sv["score_lof"], st["score_lof"] = lof_score(Xva), lof_score(Xte)
    # 5) Autoencodeur (MLP de reconstruction, sans étiquettes)
    _log("Autoencodeur...")
    sub = rng.choice(len(Xtr), size=min(C.AE_FIT_SIZE, len(Xtr)), replace=False)
    ae = MLPRegressor(hidden_layer_sizes=(16, 6, 16), max_iter=40, early_stopping=True, random_state=C.RANDOM_STATE)
    ae.fit(z(Xtr.iloc[sub]), z(Xtr.iloc[sub]))
    ae_score = lambda X: ((z(X) - ae.predict(z(X))) ** 2).mean(axis=1)
    sv["score_ae"], st["score_ae"] = ae_score(Xva), ae_score(Xte)

    _log("Seuil par coût métier (validation) et évaluation (test)...")
    amt_va, amt_te = va["amt"].to_numpy(), te["amt"].to_numpy()
    models_out = {}
    for name, col in MODELS.items():
        thr, cost_val = best_threshold(yva, sv[col], amt_va, C.REVIEW_COST)
        models_out[name] = {"col": col, "type": TYPES[col], "threshold": None if np.isinf(thr) else thr,
                            "val": evaluate(yva, sv[col], amt_va, thr, k_val, C.REVIEW_COST),
                            "test": evaluate(yte, st[col], amt_te, thr, k_te, C.REVIEW_COST)}
        t_ = models_out[name]["test"]
        _log(f"  {name:32s} PR-AUC test={t_['pr_auc']:.4f}  P@k={t_['precision_at_k']:.3f}  coût test={t_['cost']:,.0f}")

    best_sup = max((n for n in MODELS if TYPES[MODELS[n]] == "Supervisé"), key=lambda n: models_out[n]["val"]["pr_auc"])
    meta = {"features": FEATURES, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "period": {n: [str(d[C.TIME_COL].min()), str(d[C.TIME_COL].max())] for n, d in (("train", tr), ("val", va), ("test", te))},
            "fraud_rate": {"train": float(ytr.mean()), "val": float(yva.mean()), "test": float(yte.mean())},
            "budget_frac": C.BUDGET_FRAC, "k_val": k_val, "k_test": k_te, "review_cost": C.REVIEW_COST,
            "imbalance_best": imb_best, "best_supervised": best_sup,
            "behavioral_defaults": {f: float(Xtr[f].median()) for f in ("n_tx_1h", "n_tx_24h", "amt_ratio_30d", "log_gap_h")},
            "explained_model": "XGBoost"}
    json.dump({"meta": meta, "imbalance": imb_rows, "models": models_out, "eda": eda},
              open(C.ART_DIR / "results.json", "w"), indent=1, ensure_ascii=False)
    joblib.dump(xgb_final, C.ART_DIR / "xgb_final.joblib")

    # Scores sauvegardés pour l'application (curseurs en direct)
    keep = [C.ID_COL, C.TIME_COL, C.TARGET, "amt", "category"]
    out_te = te[keep + FEATURES].reset_index(drop=True)
    for col in MODELS.values(): out_te[col] = st[col].astype("float32")
    out_va = va[keep].reset_index(drop=True)
    for col in MODELS.values(): out_va[col] = sv[col].astype("float32")
    out_te.to_parquet(C.ART_DIR / "scored_test.parquet"); out_va.to_parquet(C.ART_DIR / "scored_val.parquet")

    _log("Figures pour le rapport...")
    plt.figure(figsize=(7, 5))
    for name, col in MODELS.items():
        p, r, _ = precision_recall_curve(yte, st[col]); plt.plot(r, p, label=f"{name} (AP={models_out[name]['test']['pr_auc']:.3f})")
    plt.axhline(yte.mean(), ls=":", c="gray", label="hasard"); plt.xlabel("Rappel"); plt.ylabel("Précision")
    plt.title("Courbes précision-rappel (test)"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(C.FIG_DIR / "pr_curves.png", dpi=150); plt.close()
    thr_c, cost_c = cost_curve(yva, sv["score_xgb"], amt_va, C.REVIEW_COST)
    plt.figure(figsize=(7, 4)); plt.plot(thr_c, cost_c); plt.xlabel("Seuil (XGBoost)"); plt.ylabel("Coût validation ($)")
    th = models_out["XGBoost"]["threshold"]
    if th is not None: plt.axvline(th, c="r", ls="--", label=f"seuil optimal = {th:.3f}"); plt.legend()
    plt.title("Coût métier en fonction du seuil"); plt.tight_layout(); plt.savefig(C.FIG_DIR / "cost_curve.png", dpi=150); plt.close()
    import shap
    samp = Xte.sample(min(C.SHAP_SAMPLE, len(Xte)), random_state=C.RANDOM_STATE)
    sh, _ = shap_values(xgb_final, samp)
    shap.summary_plot(sh, samp, show=False, max_display=15); plt.tight_layout()
    plt.savefig(C.FIG_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()
    _log("Terminé. Artefacts dans artifacts/")
    return meta

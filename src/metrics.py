"""Métriques pour événements rares, budget d'alertes et coût métier (montants)."""
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, fbeta_score,
                             matthews_corrcoef)


def precision_recall_at_k(y, score, k):
    k = max(1, int(k))
    top = np.argpartition(-score, min(k, len(score)) - 1)[:k]
    tp = y[top].sum()
    return tp / k, tp / max(y.sum(), 1)


def business_cost(y, amt, alert, review_cost):
    """Coût = montants des fraudes manquées + coût d'analyse des alertes."""
    alert = alert.astype(bool)
    missed = float(amt[(y == 1) & ~alert].sum())
    review = float(review_cost * alert.sum())
    return missed + review, missed, review


def best_threshold(y, score, amt, review_cost):
    """Seuil minimisant le coût (balayage exact sur les scores triés). Retourne (seuil, coût)."""
    order = np.argsort(-score, kind="stable")
    s, fa = score[order], (amt * (y == 1))[order]
    total = fa.sum()
    j = np.arange(0, len(s) + 1)
    cost = total - np.concatenate([[0.0], np.cumsum(fa)]) + review_cost * j
    # on ne coupe qu'entre deux scores distincts
    valid_cut = np.concatenate([[True], (s[:-1] != s[1:]), [True]])
    cost_v = np.where(valid_cut, cost, np.inf)
    jb = int(np.argmin(cost_v))
    thr = np.inf if jb == 0 else float(s[jb - 1])
    return thr, float(cost[jb])


def cost_curve(y, score, amt, review_cost, n_points=300):
    order = np.argsort(-score, kind="stable")
    s, fa = score[order], (amt * (y == 1))[order]
    cost = fa.sum() - np.concatenate([[0.0], np.cumsum(fa)]) + review_cost * np.arange(len(s) + 1)
    idx = np.unique(np.linspace(1, len(s), n_points).astype(int))
    return s[idx - 1], cost[idx]


def evaluate(y, score, amt, thr, k, review_cost):
    alert = score >= thr
    p_k, r_k = precision_recall_at_k(y, score, k)
    cost, missed, review = business_cost(y, amt, alert, review_cost)
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "precision_at_k": float(p_k), "recall_at_k": float(r_k),
        "n_alerts": int(alert.sum()),
        "f2": float(fbeta_score(y, alert, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, alert)) if alert.any() else 0.0,
        "cost": cost, "missed_amt": missed, "review_cost": review,
        "cost_no_model": float(amt[y == 1].sum()),
    }

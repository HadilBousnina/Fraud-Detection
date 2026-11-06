"""Application web interactive — détection de fraude explicable (Sparkov).
Lancer :  streamlit run app/streamlit_app.py     (après python run_pipeline.py)
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import fbeta_score, matthews_corrcoef, precision_recall_curve

from src import config as C
from src.explain import reason_codes, shap_values
from src.features import FEATURES
from src.metrics import cost_curve

st.set_page_config(page_title="Détection de fraude explicable", page_icon="🛡️", layout="wide")

if not (C.ART_DIR / "results.json").exists():
    st.error("Artefacts introuvables. Exécutez d'abord : `python run_pipeline.py`")
    st.stop()


@st.cache_data
def load():
    res = json.load(open(C.ART_DIR / "results.json"))
    return res, pd.read_parquet(C.ART_DIR / "scored_val.parquet"), pd.read_parquet(C.ART_DIR / "scored_test.parquet")


@st.cache_resource
def load_model():
    return joblib.load(C.ART_DIR / "xgb_final.joblib")


@st.cache_data
def global_shap(n=C.SHAP_SAMPLE):
    _, _, te = load()
    X = te[FEATURES].sample(min(n, len(te)), random_state=C.RANDOM_STATE)
    sh, base = shap_values(load_model(), X)
    return X, sh


res, val, test = load()
meta, eda, models = res["meta"], res["eda"], res["models"]
MCOL = {n: m["col"] for n, m in models.items()}

page = st.sidebar.radio("Navigation", ["1 · Données", "2 · Modèles", "3 · Seuil & budget", "4 · Explications"])
st.sidebar.caption("Jeu : Credit Card Transactions (Sparkov, synthétique)\n\nSplit chronologique 60/20/20")
st.title("🛡️ Détection de fraude explicable")

# ------------------------------------------------------------------ 1. DONNÉES
if page.startswith("1"):
    c = st.columns(4)
    c[0].metric("Transactions", f"{eda['n']:,}")
    c[1].metric("Fraudes", f"{eda['n_fraud']:,}")
    c[2].metric("Taux de fraude", f"{eda['fraud_rate']:.3%}")
    c[3].metric("Période", f"{eda['date_min'][:10]} → {eda['date_max'][:10]}")
    h = eda["amt_hist"]; mid = [(a + b) / 2 for a, b in zip(h["edges"][:-1], h["edges"][1:])]
    fig = go.Figure()
    fig.add_bar(x=mid, y=h["legit"], name="Légitime", opacity=.6); fig.add_bar(x=mid, y=h["fraud"], name="Fraude", opacity=.6)
    fig.update_layout(barmode="overlay", title="Distribution des montants par classe (densité, échelle log10)",
                      xaxis_title="log10(montant + 1)", yaxis_title="Densité")
    st.plotly_chart(fig)
    st.caption(f"Montant médian : légitime {eda['amt_median']['legit']:.1f} $ vs fraude {eda['amt_median']['fraud']:.1f} $")

    def bar(key, title, xlab):
        d = eda[key]; f = px.bar(x=d["index"], y=[r * 100 for r in d["rate"]], title=title,
                                 labels={"x": xlab, "y": "Taux de fraude (%)"}); return f
    st.plotly_chart(bar("by_month", "Évolution du taux de fraude dans le temps (par mois)", "Mois"))
    a, b = st.columns(2)
    a.plotly_chart(bar("by_hour", "Taux de fraude par heure", "Heure"))
    b.plotly_chart(bar("by_dow", "Taux de fraude par jour (0 = lundi)", "Jour"))
    st.plotly_chart(bar("by_category", "Taux de fraude par catégorie de marchand", "Catégorie"))
    st.info(f"Découpage chronologique — train {meta['period']['train'][0][:10]} → {meta['period']['train'][1][:10]} · "
            f"val → {meta['period']['val'][1][:10]} · test → {meta['period']['test'][1][:10]}. "
            f"Stratégie anti-déséquilibre retenue (comparée en validation) : **{meta['imbalance_best']}**.")
    st.subheader("Comparaison des stratégies contre le déséquilibre (validation, XGBoost)")
    st.dataframe(pd.DataFrame(res["imbalance"]).rename(columns={
        "strategie": "Stratégie", "n_train": "Lignes d'entraînement", "pr_auc_val": "PR-AUC val",
        "precision_at_k_val": "Précision@k val", "recall_at_k_val": "Rappel@k val", "temps_s": "Temps (s)"}), hide_index=True)

# ------------------------------------------------------------------ 2. MODÈLES
elif page.startswith("2"):
    st.subheader("Tableau comparatif (seuil choisi sur la validation, évalué sur le test)")
    rows = []
    for n, m in models.items():
        t = m["test"]
        rows.append({"Modèle": n, "Type": m["type"], "PR-AUC val": m["val"]["pr_auc"], "PR-AUC test": t["pr_auc"],
                     "ROC-AUC test": t["roc_auc"], f"Précision@{meta['k_test']}": t["precision_at_k"],
                     f"Rappel@{meta['k_test']}": t["recall_at_k"], "F2": t["f2"], "MCC": t["mcc"],
                     "Alertes": t["n_alerts"], "Coût test ($)": t["cost"],
                     "Gain vs sans modèle": 1 - t["cost"] / t["cost_no_model"] if t["cost_no_model"] else 0})
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.3f}" for c in ["PR-AUC val", "PR-AUC test", "ROC-AUC test", "F2", "MCC"]} |
                 {c: "{:.1%}" for c in rows[0] if c.startswith(("Précision@", "Rappel@", "Gain"))} | {"Coût test ($)": "{:,.0f}"}),
                 hide_index=True)
    st.caption(f"Budget k = {meta['budget_frac']:.1%} des transactions ({meta['k_test']:,} alertes sur le test) · "
               f"coût d'analyse = {meta['review_cost']} $ par alerte · coût = montants des fraudes manquées + analyses.")
    fig = go.Figure(); y = test[C.TARGET].to_numpy()
    for n, col in MCOL.items():
        p, r, _ = precision_recall_curve(y, test[col].to_numpy()); step = max(1, len(p) // 1500)
        fig.add_scatter(x=r[::step], y=p[::step], mode="lines", name=n)
    fig.update_layout(title="Courbes précision-rappel superposées (test)", xaxis_title="Rappel", yaxis_title="Précision")
    st.plotly_chart(fig)

# ------------------------------------------------------------------ 3. SEUIL & BUDGET
elif page.startswith("3"):
    c1, c2, c3 = st.columns(3)
    name = c1.selectbox("Modèle", list(models), index=list(models).index(meta["best_supervised"]))
    split = c2.radio("Jeu", ["Test", "Validation"], horizontal=True)
    review = c3.number_input("Coût d'analyse d'une alerte ($)", 0.0, 500.0, float(meta["review_cost"]), 0.5)
    d = test if split == "Test" else val
    col = MCOL[name]; s = d[col].to_numpy(np.float64); y = d[C.TARGET].to_numpy(); amt = d["amt"].to_numpy()
    smin, smax = float(s.min()), float(s.max())
    thr0 = models[name]["threshold"]; thr0 = smax if thr0 is None else float(np.clip(thr0, smin, smax))
    a, b = st.columns(2)
    thr = a.slider("Seuil d'alerte", smin, smax, thr0, (smax - smin) / 1000, format="%.4f")
    budget = b.slider("Budget d'alertes (% des transactions)", 0.05, 5.0, meta["budget_frac"] * 100, 0.05)
    use_budget = b.checkbox("Appliquer le budget (plafonner le nombre d'alertes)", True)
    k = int(budget / 100 * len(d))
    alert = s >= thr
    if use_budget and alert.sum() > k:                       # on garde les k scores les plus élevés
        cutoff = np.partition(s, -k)[-k]; alert = alert & (s >= cutoff)
    tp = int((alert & (y == 1)).sum()); n_al = int(alert.sum()); nf = int(y.sum())
    cost = amt[(y == 1) & ~alert].sum() + review * n_al; cost0 = amt[y == 1].sum()
    m = st.columns(6)
    m[0].metric("Alertes", f"{n_al:,}"); m[1].metric("Précision", f"{tp / max(n_al, 1):.1%}")
    m[2].metric("Rappel", f"{tp / max(nf, 1):.1%}"); m[3].metric("F2", f"{fbeta_score(y, alert, beta=2, zero_division=0):.3f}")
    m[4].metric("MCC", f"{matthews_corrcoef(y, alert) if alert.any() else 0:.3f}")
    m[5].metric("Coût total", f"{cost:,.0f} $", f"{(cost - cost0) / max(cost0, 1):+.1%} vs sans modèle", delta_color="inverse")
    st.caption(f"Fraudes détectées : {tp}/{nf} · montants récupérés : {amt[alert & (y == 1)].sum():,.0f} $ · "
               f"fraudes manquées : {amt[(y == 1) & ~alert].sum():,.0f} $ · analyses : {review * n_al:,.0f} $")
    x, cst = cost_curve(y, s, amt, review)
    fig = go.Figure(); fig.add_scatter(x=x, y=cst, mode="lines", name="Coût")
    fig.add_vline(x=thr, line_dash="dash", annotation_text="seuil courant")
    fig.update_layout(title="Coût total en fonction du seuil", xaxis_title="Seuil", yaxis_title="Coût ($)")
    st.plotly_chart(fig)
    st.caption("Le seuil par défaut est celui qui minimise le coût sur la **validation** ; basculez sur « Test » pour le confirmer.")

# ------------------------------------------------------------------ 4. EXPLICATIONS
else:
    model = load_model()
    st.subheader("Explication globale (SHAP — XGBoost)")
    X, sh = global_shap()
    shap.summary_plot(sh, X, show=False, max_display=15); st.pyplot(plt.gcf(), clear_figure=True)
    st.subheader("Explication d'une alerte")
    thr = models["XGBoost"]["threshold"] or 1.0
    top = test.assign(_p=test["score_xgb"]).sort_values("_p", ascending=False).head(200).reset_index()
    labels = [f"{r[C.ID_COL][:8]} · {r['amt']:.0f} $ · {r['category']} · p={r['_p']:.2f}" + (" ✅ fraude" if r[C.TARGET] == 1 else "")
              for _, r in top.iterrows()]
    pick = st.selectbox("Transaction (200 scores les plus élevés du test)", range(len(top)), format_func=lambda i: labels[i])
    row = top.iloc[pick]; xr = pd.DataFrame([row[FEATURES].astype("float32")])
    sh1, base = shap_values(model, xr)
    st.write(f"**Probabilité de fraude : {row['_p']:.1%}** · seuil d'alerte : {thr:.3f} · "
             f"{'🚨 ALERTE' if row['_p'] >= thr else 'pas d\u2019alerte'} · vérité terrain : {'fraude' if row[C.TARGET] == 1 else 'légitime'}")
    st.markdown("**Codes motifs** (variables ayant le plus contribué à l'alerte)")
    st.table(pd.DataFrame(reason_codes(sh1[0], xr.iloc[0], top=5)).rename(
        columns={"motif": "Motif", "valeur": "Valeur", "contribution_shap": "Contribution (log-odds)"})[["Motif", "Valeur", "Contribution (log-odds)"]])
    shap.plots.waterfall(shap.Explanation(values=sh1[0], base_values=base, data=xr.iloc[0].values,
                                          feature_names=FEATURES), max_display=12, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

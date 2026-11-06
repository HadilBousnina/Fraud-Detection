# Challenge 2 — Détection de fraude explicable (Sparkov)

Data Mining · II3-IA · ENSI · 2026-2027 — Jeu : **Credit Card Transactions Fraud Detection (simulateur Sparkov)**, Kaggle `kartik2112/fraud-detection` (synthétique, ~1,85 M transactions, ~0,58 % de fraudes). Vérifiez la licence sur la page du jeu.

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt

# 1) Données : une des 3 options
#    a) automatique (kagglehub, nécessite un compte/token Kaggle)
#    b) copier fraudTrain.csv et fraudTest.csv dans ./data
#    c) notebook Kaggle : ajouter le dataset en entrée, il est trouvé dans /kaggle/input

# 2) Pipeline complet (variables, split, modèles, seuil, figures, artefacts)
python run_pipeline.py                     # ou --max-rows 300000 pour un essai rapide

# 3) Application web (Modules 1 à 5)
streamlit run app/streamlit_app.py

# 4) API (Module 6)  →  http://127.0.0.1:8000/docs
uvicorn api.main:app --reload

# Tests unitaires (anti-fuite, split, coût, précision@k)
pytest -q
```

Sur Kaggle : `!pip install -q xgboost shap` puis `!python run_pipeline.py --data /kaggle/input`.
Durée indicative sur le jeu complet : quelques minutes (LOF et l'autoencodeur utilisent des sous-échantillons, voir `src/config.py`).

## Correspondance avec le cahier des charges

| Module | Où |
|---|---|
| 1 Données (taux de fraude, montants par classe, évolution, heure/catégorie) | page « 1 · Données », `make_eda` dans `src/pipeline.py` |
| 2 Variables (heure cyclique, jour, vélocité 1 h / 24 h, montant vs moyenne 30 j, délai depuis la précédente) | `src/features.py` |
| 2 Split chronologique 60/20/20 | `chronological_split` dans `src/data.py` |
| 2 Déséquilibre (train seul, comparé en validation) | `compare_imbalance` : aucune / pondération / sous-échantillonnage 10:1 |
| 3 Modèles : supervisés (régression logistique pondérée, XGBoost) ; non supervisés (Isolation Forest, LOF, autoencodeur) + tableau comparatif | `src/pipeline.py`, page « 2 · Modèles » |
| 4 PR-AUC, ROC-AUC, précision@k, rappel@k, F2, MCC ; courbes PR superposées | `src/metrics.py`, page « 2 · Modèles » |
| 4 Seuil par coût métier (validation → confirmé sur test) ; curseurs seuil + budget en direct | `best_threshold`, page « 3 · Seuil & budget » |
| 5 SHAP global (beeswarm), waterfall d'une alerte, codes motifs | `src/explain.py`, page « 4 · Explications » |
| 6 API FastAPI : `/predict`, `/explain`, `/predict/batch`, `/model-info`, `/health` | `api/main.py` |

> ⚠️ Le PDF fourni est tronqué au Module 6 (le texte s'arrête au titre). L'API ci-dessus est une version standard : comparez avec l'énoncé complet de Mme Besrour et ajustez si des endpoints précis sont exigés.

## Choix méthodologiques (à reprendre et justifier dans le rapport)

- **Sans fuite** : les variables de vélocité et de profil n'utilisent que les transactions *strictement antérieures* de la même carte (testé dans `tests/test_core.py`). Les identifiants (`cc_num`, nom, `trans_num`) ne sont jamais des variables du modèle.
- **Chronologie** : les fichiers `fraudTrain` (2019 → mi-2020) et `fraudTest` (mi-2020 → fin 2020) sont fusionnés, triés par date, puis re-découpés en 60/20/20.
- **Coût métier** : coût = Σ montants des fraudes manquées + `REVIEW_COST` × nombre d'alertes. Hypothèses modifiables dans `src/config.py` : `REVIEW_COST = 5 $`, budget `k = 1 %` des transactions (capacité d'analyse). **Ce sont des hypothèses : justifiez-les.**
- **Seuil** : choisi sur la validation (balayage exact), puis évalué sur le test sans réajustement.
- **Modèle expliqué** : XGBoost (valeurs SHAP exactes via `pred_contribs`).
- **Limites à discuter** : données synthétiques (les motifs de fraude sont ceux du simulateur, les performances seront plus optimistes que sur des données réelles) ; coût d'analyse supposé constant ; pas de feedback loop ; dérive temporelle.

## Arborescence

```
run_pipeline.py        # point d'entrée du pipeline
src/                   # config, data, features, metrics, explain, pipeline
app/streamlit_app.py   # application web
api/main.py            # API FastAPI
tests/test_core.py     # tests unitaires
scripts/make_synthetic.py   # mini-jeu factice pour tester le code SANS Kaggle (ne pas utiliser pour le rapport)
artifacts/             # généré : results.json, modèle, scores, figures (PR, coût, SHAP)
```

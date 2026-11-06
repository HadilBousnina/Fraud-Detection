"""Configuration centrale du projet (modifiable sans toucher au reste du code)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
FIG_DIR = ART_DIR / "figures"

# --- Jeu de données : Credit Card Transactions Fraud Detection (Sparkov) ---
KAGGLE_HANDLE = "kartik2112/fraud-detection"   # fraudTrain.csv + fraudTest.csv
TARGET = "is_fraud"
TIME_COL = "trans_date_trans_time"
CARD_COL = "cc_num"
ID_COL = "trans_num"

# --- Découpage chronologique 60 / 20 / 20 ---
SPLIT = (0.6, 0.2, 0.2)

# --- Hypothèses métier (à justifier dans le rapport) ---
BUDGET_FRAC = 0.01     # budget d'alertes : 1 % des transactions sont examinées
REVIEW_COST = 5.0      # coût d'analyse d'une alerte (en $, temps analyste)

RANDOM_STATE = 42

# Taille d'échantillon pour les méthodes lentes
LOF_FIT_SIZE = 20_000
AE_FIT_SIZE = 200_000
SHAP_SAMPLE = 2_000

"""Lance tout le pipeline. Exemples :
    python run_pipeline.py                      # téléchargement auto via kagglehub (ou ./data, ou /kaggle/input)
    python run_pipeline.py --data ./data        # dossier contenant fraudTrain.csv et fraudTest.csv
    python run_pipeline.py --max-rows 300000    # essai rapide
"""
import argparse
from src.pipeline import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    a = ap.parse_args()
    run(a.data, a.max_rows)

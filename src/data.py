"""Chargement du jeu Sparkov (Kaggle: kartik2112/fraud-detection).

Ordre de recherche des fichiers fraudTrain.csv / fraudTest.csv :
  1. chemin fourni (--data), 2. dossier ./data, 3. /kaggle/input (notebook Kaggle),
  4. téléchargement automatique via kagglehub.

Le découpage Train/Test d'origine est déjà chronologique ; on les concatène puis
on retrie par date pour refaire notre propre découpage 60/20/20.
"""
from pathlib import Path
import pandas as pd
from .config import DATA_DIR, KAGGLE_HANDLE, TIME_COL


def _find_csvs(root: Path):
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*.csv") if p.name in ("fraudTrain.csv", "fraudTest.csv"))


def locate_files(path=None):
    candidates = [path] if path else []
    candidates += [DATA_DIR, "/kaggle/input"]
    for c in candidates:
        if c and Path(c).exists():
            files = _find_csvs(c)
            if files:
                return files
    import kagglehub  # téléchargement automatique
    folder = kagglehub.dataset_download(KAGGLE_HANDLE)
    files = _find_csvs(folder)
    if not files:
        raise FileNotFoundError(f"fraudTrain.csv / fraudTest.csv introuvables dans {folder}")
    return files


def load_raw(path=None, max_rows=None) -> pd.DataFrame:
    files = locate_files(path)
    print("Fichiers :", [str(f) for f in files])
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    if max_rows:
        df = df.iloc[:max_rows].copy()   # les max_rows premières transactions (ordre chronologique)
    return df


def chronological_split(df: pd.DataFrame, frac=(0.6, 0.2, 0.2)):
    """Split chronologique : le futur n'est jamais utilisé pour apprendre le passé."""
    n = len(df)
    i1, i2 = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()

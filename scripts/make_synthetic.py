"""Génère un MINI jeu synthétique au format Sparkov, uniquement pour tester le code sans Kaggle.
(Les vrais résultats doivent être produits avec le vrai jeu.)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from src.features import CATEGORIES

def make(n_cards=150, n=60_000, seed=0):
    rng = np.random.RandomState(seed)
    cards = pd.DataFrame({"cc_num": rng.randint(10**15, 10**16, n_cards, dtype=np.int64),
                          "gender": rng.choice(["F", "M"], n_cards), "lat": rng.uniform(25, 48, n_cards),
                          "long": rng.uniform(-120, -70, n_cards), "city_pop": rng.randint(500, 2_000_000, n_cards),
                          "dob": pd.to_datetime("1940-01-01") + pd.to_timedelta(rng.randint(0, 20000, n_cards), "D")})
    t0 = pd.Timestamp("2019-01-01")
    ts = t0 + pd.to_timedelta(np.sort(rng.randint(0, 365 * 24 * 3600, n)), "s")
    df = cards.iloc[rng.randint(0, n_cards, n)].reset_index(drop=True)
    df["trans_date_trans_time"] = ts
    df["category"] = rng.choice(CATEGORIES, n)
    df["amt"] = np.round(np.exp(rng.normal(3.5, 1.0, n)), 2)
    df["merch_lat"] = df.lat + rng.normal(0, 0.4, n); df["merch_long"] = df.long + rng.normal(0, 0.4, n)
    hour = ts.hour
    logit = -7 + 1.8 * (df.category.isin(["shopping_net", "misc_net", "grocery_pos"])) + 2.0 * ((hour >= 22) | (hour < 4)) + 0.004 * df.amt
    fraud = rng.rand(n) < 1 / (1 + np.exp(-logit))
    df.loc[fraud, "amt"] = np.round(df.loc[fraud, "amt"] * rng.uniform(2, 6, fraud.sum()), 2)
    df["is_fraud"] = fraud.astype(int)
    df["trans_num"] = [f"{i:032x}" for i in range(n)]
    df["unix_time"] = ts.astype("int64") // 10**9
    return df

if __name__ == "__main__":
    out = Path("data_synth"); out.mkdir(exist_ok=True)
    df = make(); k = int(len(df) * 0.7)
    df.iloc[:k].to_csv(out / "fraudTrain.csv"); df.iloc[k:].to_csv(out / "fraudTest.csv")
    print("fraude:", df.is_fraud.mean(), "->", out)

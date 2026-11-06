"""Variables temporelles, géographiques et comportementales (sans fuite de données).

Règle anti-fuite : toute variable de vélocité / profil est calculée UNIQUEMENT avec
les transactions strictement antérieures de la même carte (la transaction courante
et le futur sont exclus).
"""
import numpy as np
import pandas as pd
from .config import CARD_COL, TIME_COL

CATEGORIES = ["entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
              "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
              "shopping_net", "shopping_pos", "travel"]

BEHAVIORAL = ["n_tx_1h", "n_tx_24h", "amt_ratio_30d", "log_gap_h"]
STATELESS = (["log_amt", "hour_sin", "hour_cos", "dow", "is_weekend", "is_night",
              "dist_km", "age", "log_city_pop", "gender_m"] + [f"cat_{c}" for c in CATEGORIES])
FEATURES = STATELESS + BEHAVIORAL

# Libellés lisibles pour les « codes motifs »
LABELS = {
    "log_amt": "Montant de la transaction", "hour_sin": "Heure (cycle)", "hour_cos": "Heure (cycle)",
    "dow": "Jour de la semaine", "is_weekend": "Week-end", "is_night": "Transaction de nuit",
    "dist_km": "Distance client – marchand", "age": "Âge du client", "log_city_pop": "Taille de la ville",
    "gender_m": "Genre", "n_tx_1h": "Nb de transactions sur 1 h", "n_tx_24h": "Nb de transactions sur 24 h",
    "amt_ratio_30d": "Montant vs moyenne des 30 derniers jours", "log_gap_h": "Délai depuis la transaction précédente",
    **{f"cat_{c}": f"Catégorie « {c} »" for c in CATEGORIES},
}


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def add_stateless_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t = pd.to_datetime(df[TIME_COL])
    hour = t.dt.hour + t.dt.minute / 60.0
    df["hour"] = hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)          # codage cyclique de l'heure
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow"] = t.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_night"] = ((hour < 6) | (hour >= 22)).astype(int)
    df["log_amt"] = np.log1p(df["amt"])
    df["dist_km"] = haversine(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    df["age"] = (t - pd.to_datetime(df["dob"])).dt.days / 365.25
    df["log_city_pop"] = np.log1p(df["city_pop"])
    df["gender_m"] = (df["gender"] == "M").astype(int)
    for c in CATEGORIES:
        df[f"cat_{c}"] = (df["category"] == c).astype(int)
    return df


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """df doit être trié par date. Calcul vectorisé par carte (np.searchsorted)."""
    df = df.copy()
    t = pd.to_datetime(df[TIME_COL]).to_numpy().astype("datetime64[s]").astype("int64")
    amt = df["amt"].to_numpy(dtype=float)
    n = len(df)
    n1, n24 = np.zeros(n), np.zeros(n)
    ratio, gap = np.ones(n), np.full(n, 720.0)   # 720 h = 30 j : « pas d'historique »
    for _, idx in df.groupby(CARD_COL).indices.items():
        ts, a = t[idx], amt[idx]
        pos = np.arange(len(idx))
        n1[idx] = pos - np.searchsorted(ts, ts - 3600, side="left")        # passé uniquement
        n24[idx] = pos - np.searchsorted(ts, ts - 86400, side="left")
        lo30 = np.searchsorted(ts, ts - 30 * 86400, side="left")
        cs = np.concatenate([[0.0], np.cumsum(a)])
        cnt = pos - lo30
        mean_prev = np.divide(cs[pos] - cs[lo30], cnt, out=np.full(len(idx), np.nan), where=cnt > 0)
        r = a / mean_prev
        ratio[idx] = np.where(cnt > 0, r, 1.0)
        if len(idx) > 1:
            g = np.empty(len(idx)); g[0] = 720.0
            g[1:] = np.minimum((ts[1:] - ts[:-1]) / 3600.0, 720.0)
            gap[idx] = g
    df["n_tx_1h"], df["n_tx_24h"] = n1, n24
    df["amt_ratio_30d"] = np.clip(ratio, 0, 50)
    df["log_gap_h"] = np.log1p(gap)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_behavioral_features(add_stateless_features(df))

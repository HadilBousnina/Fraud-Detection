import numpy as np, pandas as pd, pytest
from src.features import add_behavioral_features, add_stateless_features
from src.metrics import best_threshold, business_cost, precision_recall_at_k
from src.data import chronological_split


def _df():
    t = pd.to_datetime(["2020-01-01 10:00", "2020-01-01 10:30", "2020-01-01 12:00", "2020-01-05 09:00"])
    return pd.DataFrame({"trans_date_trans_time": t, "cc_num": 1, "amt": [10., 20., 30., 60.]})


def test_velocity_uses_only_past():
    d = add_behavioral_features(_df())
    assert d.n_tx_1h.tolist() == [0, 1, 0, 0]       # la transaction courante n'est jamais comptée
    assert d.n_tx_24h.tolist() == [0, 1, 2, 0]
    assert d.amt_ratio_30d.iloc[2] == pytest.approx(30 / 15)   # moyenne des 2 transactions passées
    assert d.amt_ratio_30d.iloc[0] == 1.0                      # pas d'historique -> neutre


def test_no_future_leak():
    d = _df(); full = add_behavioral_features(d)
    cut = add_behavioral_features(d.iloc[:3])
    assert full.iloc[:3][["n_tx_1h", "n_tx_24h", "amt_ratio_30d"]].equals(cut[["n_tx_1h", "n_tx_24h", "amt_ratio_30d"]])


def test_chrono_split_order():
    d = pd.DataFrame({"trans_date_trans_time": pd.date_range("2020", periods=100, freq="h")})
    a, b, c = chronological_split(d)
    assert (len(a), len(b), len(c)) == (60, 20, 20)
    assert a.trans_date_trans_time.max() < b.trans_date_trans_time.min() < c.trans_date_trans_time.min()


def test_cost_and_threshold():
    y = np.array([1, 0, 1, 0, 0]); amt = np.array([100., 5, 50, 5, 5]); s = np.array([.9, .8, .3, .2, .1])
    cost, missed, review = business_cost(y, amt, s >= .5, 5.0)
    assert (missed, review) == (50.0, 10.0)
    thr, c = best_threshold(y, s, amt, 5.0)
    assert c == min(business_cost(y, amt, s >= t, 5.0)[0] for t in [0.0, .1, .2, .3, .8, .9, 1.1])


def test_precision_at_k():
    y = np.array([1, 0, 1, 0]); s = np.array([.9, .8, .7, .1])
    assert precision_recall_at_k(y, s, 2) == (0.5, 0.5)

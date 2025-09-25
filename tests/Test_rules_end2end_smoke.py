# tests/test_rules_end2end_smoke.py
from src.rules import prepare_aux_cols, predict_pairs
from src.cluster import build_clusters
import pandas as pd

def test_end2end_rules_smoke(small_df):
    df = prepare_aux_cols(small_df.copy())
    cand = {(0,1), (0,2), (1,2)}
    pred = predict_pairs(df, cand)
    clusters = build_clusters(pred, df.index)
    # хотя бы одна компонента больше 1 (дубликаты 0,1)
    assert any(len(c) > 1 for c in clusters)

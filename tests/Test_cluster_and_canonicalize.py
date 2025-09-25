# tests/test_cluster_and_canonicalize.py
import pandas as pd
from src.rules import prepare_aux_cols
from src.cluster import build_clusters, summarize_clusters
from src.canonicalize import canonicalize_all

def test_build_clusters_and_summary(small_df):
    df = prepare_aux_cols(small_df.copy())
    # два ребра: один кластер [0,1], другой одиночка [2]
    pairs = {(0, 1)}
    clusters = build_clusters(pairs, df.index)
    # ожидаем 2 компоненты связности
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]

    clust_df = summarize_clusters(df, clusters)
    # проверим сводку по размерам
    assert set(clust_df["size"]) == {1, 2}
    # топ-uid доля для кластера размера 2 должна быть 1.0 (оба имеют один и тот же uid=1)
    row = clust_df.loc[clust_df["size"] == 2].iloc[0]
    assert abs(float(row["top_uid_share"]) - 1.0) < 1e-9

def test_canonicalize_all_simple(small_df, canon_rules):
    df = prepare_aux_cols(small_df.copy())
    clusters = [[0,1], [2]]  # руками
    _, entities = canonicalize_all(df, clusters, canon_rules)

    # Должно быть 2 сущности
    assert len(entities) == 2

    # Для сущности из [0,1] имя должно быть самым длинным "john doe"
    e0 = entities.sort_values("entity_id").iloc[0]
    assert "john doe" in e0["Name_norm"].lower()

    # zip берём по большинству
    assert e0["Zip_norm"] == "12345"

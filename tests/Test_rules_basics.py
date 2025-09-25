# tests/test_rules_basics.py
import math
import pandas as pd
from src.rules import prepare_aux_cols, pair_features, is_match, predict_pairs

def test_pair_features_keys_and_types(small_df):
    df = prepare_aux_cols(small_df.copy())
    f = pair_features(df, 0, 1)

    # ожидаемые признаки
    expected = {
        "name_sim", "street_sim",
        "zip_eq", "city_eq",
        "email_eq", "phone_eq",
        "email_user_eq", "phone_last4_eq",
    }
    assert expected.issubset(set(f.keys()))

    # типы/диапазоны
    assert 0.0 <= f["name_sim"] <= 1.0
    assert 0.0 <= f["street_sim"] <= 100.0
    for b in ["zip_eq", "city_eq", "email_eq", "phone_eq", "email_user_eq", "phone_last4_eq"]:
        assert isinstance(f[b], (bool, int))

def test_is_match_positive_and_negative(small_df):
    df = prepare_aux_cols(small_df.copy())
    # дубль должен совпасть
    assert is_match(df, 0, 1) is True
    # разные сущности не совпадают
    assert is_match(df, 0, 2) is False

def test_predict_pairs_subset(small_df):
    df = prepare_aux_cols(small_df.copy())
    cand = {(0,1), (0,2), (1,2)}
    # Возьмём стандартные пороги из твоей реализации (функция сама знает дефолты)
    pred = predict_pairs(df, cand)
    # должно вернуть подмножество кандидатов
    assert pred.issubset(cand)
    # и точно содержит истинный дубль
    assert (0,1) in pred

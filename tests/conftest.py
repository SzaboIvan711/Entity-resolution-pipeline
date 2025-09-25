# tests/conftest.py
import pytest
import pandas as pd
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
@pytest.fixture()
def small_df():
    # два дубля (uid=1) + один другой объект (uid=2)
    df = pd.DataFrame([
        # i=0 — эталон
        dict(uid=1, Name_norm="john doe",    Street_norm="main street 10",
             City_norm="austin", Zip_norm="12345",
             Email_norm="john.doe@example.com", Phone_norm="5550012345"),
        # i=1 — тот же человек с лёгкими отличиями
        dict(uid=1, Name_norm="john d",      Street_norm="main st 10",
             City_norm="austin", Zip_norm="12345",
             Email_norm="John.Doe+promo@example.com", Phone_norm="5559912345"),  # last4 совпадают
        # i=2 — другая сущность
        dict(uid=2, Name_norm="mary smith",  Street_norm="oak ave 5",
             City_norm="dallas", Zip_norm="54321",
             Email_norm="mary@example.com",  Phone_norm="5558877000"),
    ])
    return df

@pytest.fixture()
def canon_rules():
    # под твои названия колонок
    from src.canonicalize import majority, longest, most_frequent_valid
    return {
        "Name_norm":   longest,
        "Street_norm": majority,
        "City_norm":   majority,
        "Zip_norm":    majority,
        "Email_norm":  most_frequent_valid,
        "Phone_norm":  most_frequent_valid,
    }

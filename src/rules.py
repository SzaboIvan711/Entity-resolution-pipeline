# rules.py
from typing import Dict, Tuple, Set, Iterable
from itertools import combinations
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz
import pandas as pd

# 1) Parameters (you can override thresholds via function args if needed)
NAME_THR   = 0.92
STREET_THR = 88
HARD_NAME  = 0.95

# Helper to prepare auxiliary fields (in case this wasn't done earlier)
def prepare_aux_cols(df: pd.DataFrame) -> pd.DataFrame:
    if 'email_user' not in df:
        df['email_user'] = df['Email_norm'].str.split('@').str[0]
    if 'phone_last4' not in df:
        df['phone_last4'] = df['Phone_norm'].astype(str).str[-4:]
    return df

# 2) Pairwise features
def pair_features(df: pd.DataFrame, i: int, j: int) -> Dict[str, float]:
    a, b = df.loc[i], df.loc[j]
    return {
        'name_sim': JaroWinkler.normalized_similarity(a['Name_norm'], b['Name_norm']),
        'street_sim': fuzz.token_set_ratio(a['Street_norm'], b['Street_norm']),
        'zip_eq': a['Zip_norm'] == b['Zip_norm'],
        'city_eq': ('City_norm' in df.columns and a['City_norm'] == b['City_norm']),
        'email_eq': a['Email_norm'] == b['Email_norm'],
        'phone_eq': a['Phone_norm'] == b['Phone_norm'],
        'email_user_eq': a['email_user'] == b['email_user'],
        'phone_last4_eq': a['phone_last4'] == b['phone_last4'],
    }

# 3) Rule-based matcher
def is_match(df: pd.DataFrame, i: int, j: int,
             name_thr: float = NAME_THR,
             street_thr: float = STREET_THR,
             hard_name: float = HARD_NAME) -> bool:
    f = pair_features(df, i, j)
    if f['email_eq'] or f['phone_eq']:
        return True
    if f['name_sim'] >= name_thr and (f['zip_eq'] or f['city_eq']):
        return True
    if f['street_sim'] >= street_thr and f['zip_eq']:
        return True
    if f['email_user_eq'] and (f['zip_eq'] or f['name_sim'] >= hard_name):
        return True
    if f['phone_last4_eq'] and f['name_sim'] >= hard_name and (f['zip_eq'] or f['city_eq']):
        return True
    # Safety rule for rare typos in name+street just before returning False
    if (f['zip_eq'] and f['city_eq']) and \
       (f['name_sim'] >= 0.88 and f['street_sim'] >= 82) and \
       (f['phone_last4_eq'] or f['email_user_eq']):
        return True

    return False

# --- Utilities for evaluation ---
def true_pairs(df: pd.DataFrame, uid_col: str = 'uid') -> Set[Tuple[int,int]]:
    S = set()
    for _, g in df.groupby(uid_col).groups.items():
        g = list(g)
        for i, j in combinations(sorted(g), 2):
            S.add((i, j))
    return S

def predict_pairs(df: pd.DataFrame, cand_pairs: Iterable[Tuple[int,int]],
                  **thr) -> Set[Tuple[int,int]]:
    return {p for p in cand_pairs if is_match(df, *p, **thr)}

def evaluate_pairwise(T: Set[Tuple[int,int]], P: Set[Tuple[int,int]]) -> Dict[str, float]:
    tp = len(P & T); fp = len(P - T); fn = len(T - P)
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec  = tp/(tp+fn) if (tp+fn) else 0.0
    f1   = 0 if prec+rec==0 else 2*prec*rec/(prec+rec)
    return {'precision':prec,'recall':rec,'f1':f1,'tp':tp,'fp':fp,'fn':fn,'P':len(P),'T':len(T)}

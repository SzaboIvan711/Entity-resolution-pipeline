# canonicalize.py
from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Sequence
import hashlib
import pandas as pd

# --- 1) Stable entity_id (independent of row order) ---
def stable_entity_id(idxs: Sequence[int], prefix: str = "ent_") -> str:
    key = "|".join(map(str, sorted(map(int, idxs))))
    return prefix + hashlib.sha1(key.encode()).hexdigest()[:12]

# --- 2) Strategies for choosing a canonical value ---
def majority(s: pd.Series):
    vc = s.dropna().value_counts()
    return vc.index[0] if len(vc) else None

def longest(s: pd.Series):
    s = s.dropna().astype(str)
    return max(s, key=len) if len(s) else None

def most_frequent_valid(s: pd.Series):
    # Currently identical to majority; validation for email/phone can be added here later
    return majority(s)

# --- 3) Canonicalization of a single cluster ---
def canonicalize_cluster(df: pd.DataFrame, idxs: List[int],
                         rules: Dict[str, Callable[[pd.Series], object]]) -> dict:
    d = df.loc[idxs]
    eid = stable_entity_id(idxs)

    out = {
        "entity_id": eid,
        "support_size": len(idxs),
    }

    # Core fields resolved according to provided rules
    for col, fn in rules.items():
        if col in d:
            out[col] = fn(d[col])

    # Useful metadata/aggregates
    if "Name_norm" in d:
        vc = d["Name_norm"].dropna().value_counts()
        out["name_share"] = (vc.iloc[0] / len(d)) if len(vc) else None
    if "Email_norm" in d:
        out["all_emails"] = ";".join(sorted(d["Email_norm"].dropna().unique()))
    if "Phone_norm" in d:
        out["all_phones"] = ";".join(sorted(d["Phone_norm"].dropna().unique()))

    return out

# --- 4) Canonicalization over all clusters ---
def canonicalize_all(df: pd.DataFrame, clusters: List[List[int]],
                     rules: Dict[str, Callable[[pd.Series], object]]):
    # Mapping: row_id -> entity_id
    eid_map = {}
    for idxs in clusters:
        eid = stable_entity_id(idxs)
        for i in idxs:
            eid_map[int(i)] = eid

    df_with_eid = df.copy()
    df_with_eid["entity_id"] = df_with_eid.index.map(eid_map)

    # Build the entity table
    rows = [canonicalize_cluster(df, idxs, rules) for idxs in clusters]
    entities = (pd.DataFrame(rows)
                  .sort_values(["support_size","entity_id"], ascending=[False, True])
                  .reset_index(drop=True))
    return df_with_eid, entities

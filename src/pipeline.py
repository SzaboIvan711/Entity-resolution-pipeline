# src/pipline.py
from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd


# Project-local utilities
from rules import prepare_aux_cols, pair_features, is_match  # your functions from rules.py
from cluster import build_clusters, summarize_clusters
from canonicalize import (
    canonicalize_all, majority, longest, most_frequent_valid
)

# --- Paths / constants ---
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT  = ROOT / "out"
OUT.mkdir(exist_ok=True)

CLEAR_DATA_PATH   = DATA / "clear_data.csv"
CAND_PAIRS_PATH   = OUT  / "cand_pairs.csv"      # candidate pairs after blocking
PAIRS_PRED_PATH   = OUT  / "pairs_pred.csv"      # final matched pairs (after matching)
ROWS_WITH_EID_PATH = OUT / "rows_with_entity_id.csv"
ENTITIES_PATH      = OUT / "entities.csv"

MODEL_PATH  = DATA / "pair_model.joblib"
META_PATH   = DATA / "pair_model_meta.json"  # {'features': [...], 'threshold': float}


def load_data() -> pd.DataFrame:
    """
    Read normalized data; ensure Phone_norm/Zip_norm are strings.
    Also generate auxiliary *_norm fields if required by rules/model.
    """
    df = pd.read_csv(
        CLEAR_DATA_PATH,
        dtype={"Phone_norm": str, "Zip_norm": str}
    )
    # Create auxiliary normalized fields and helper columns if needed by rules/model
    df = prepare_aux_cols(df)
    return df


def load_candidates() -> list[tuple[int, int]]:
    """
    Load candidate pairs for matching. Expects columns 'i' and 'j'
    with row indices of the source DataFrame.
    """
    cand_df = pd.read_csv(CAND_PAIRS_PATH)
    pairs = list(map(tuple, cand_df[["i", "j"]].to_numpy()))
    return pairs


# --------- Matching ---------

def predict_with_model(df: pd.DataFrame,
                       pairs: list[tuple[int, int]],
                       model_path: Path,
                       meta_path: Path) -> set[tuple[int, int]]:
    """
    Predict matches with a trained model. Supports two bundle formats:
    1) dict with keys {'clf','feat_cols','threshold'}
    2) raw estimator in joblib + external meta JSON with features/threshold
    """
    bundle = joblib.load(model_path)

    # --- Unpack model/metadata from either supported format ---
    if isinstance(bundle, dict) and "clf" in bundle:
        clf = bundle["clf"]
        feat_cols = bundle.get("feat_cols")
        thr = float(bundle.get("threshold"))
    else:
        clf = bundle
        if not meta_path.exists():
            raise RuntimeError(
                "pair_model.joblib contains only an estimator, "
                "and the meta file with features/threshold is missing: "
                f"{meta_path}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feat_cols = meta["features"]
        thr = float(meta["threshold"])

    if not feat_cols:
        raise ValueError("Failed to obtain model feature list (feat_cols).")

    # --- Build feature matrix ---
    rows, idx = [], []
    for (i, j) in pairs:
        f = pair_features(df, i, j)
        rows.append({c: f.get(c, 0) for c in feat_cols})
        idx.append((i, j))

    X = pd.DataFrame(rows)

    # Type alignment as in training notebooks
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    if "street_sim" in X.columns:
        X["street_sim"] = X["street_sim"] / 100.0  # same scale as in training

    # --- Inference ---
    proba = clf.predict_proba(X)[:, 1]
    pred = {ij for ij, p in zip(idx, proba) if p >= thr}
    return pred


def predict_with_rules(df: pd.DataFrame,
                       pairs: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """Simple baseline using the rule-based is_match()."""
    return {(i, j) for (i, j) in pairs if is_match(df, i, j)}


def match_pairs(df: pd.DataFrame, cand_pairs: list[tuple[int, int]]) -> set[tuple[int, int]]:
    """
    Auto-select matcher: use model if present, otherwise fall back to rules.
    """
    if MODEL_PATH.exists() and META_PATH.exists():
        print(f"[matching] using model: {MODEL_PATH.name}")
        return predict_with_model(df, cand_pairs, MODEL_PATH, META_PATH)
    else:
        print("[matching] using rules (fallback)")
        return predict_with_rules(df, cand_pairs)

# --------- Clustering + Canonicalization ---------

def make_entity_id(df: pd.DataFrame, pred_pairs: set[tuple[int, int]]) -> pd.Series:
    """
    Build transitive clusters from predicted pairs, then map row index -> cluster id.
    Rows not present in any pair receive entity_id = -1.
    """
    clusters = build_clusters(pred_pairs, df.index)  # -> list[list[int]]
    # index -> cluster id mapping
    eid = {}
    for cid, idxs in enumerate(clusters):
        for idx in idxs:
            eid[idx] = cid
    # if a row is a singleton (in no pairs), entity_id becomes -1
    return df.index.map(eid).fillna(-1).astype(int)


def run_canonicalization(df: pd.DataFrame, entity_id_col: str = "entity_id") -> pd.DataFrame:
    """
    Canonicalization rules for entity 'passport' fields.
    NOTE: Uses the helper available in your canonicalize.py.
    """
    canon_rules = {
        "Name_norm":  longest,              # choose the longest normalized string
        "Street_norm": majority,
        "City_norm":   majority,
        "Zip_norm":    majority,
        "Email_norm":  most_frequent_valid, # most frequent valid value
        "Phone_norm":  most_frequent_valid, # most frequent valid value
    }
    # canonicalize_all expects the source df and a clusters list; here we call the
    # convenient wrapper available in your canonicalize.py
    df_eid, entities = canonicalize_all(df, build_clusters, canon_rules, entity_id_col=entity_id_col)
    # NB: canonicalize_all in your version returns (df_eid, entities)
    return df_eid, entities


def main():
    print(">> load data")
    df = load_data()

    print(">> load candidate pairs")
    cand_pairs = load_candidates()
    print(f"candidates: {len(cand_pairs)}")

    print(">> matching")
    pred_pairs = match_pairs(df, cand_pairs)
    print(f"predicted matches: {len(pred_pairs)}")
    # save predicted pairs
    pd.DataFrame(sorted(pred_pairs), columns=["i", "j"]).to_csv(PAIRS_PRED_PATH, index=False)

    print(">> clustering")
    df["entity_id"] = make_entity_id(df, pred_pairs)
    # quick sanity metrics over clusters
    clust_df = summarize_clusters(df, build_clusters(pred_pairs, df.index))
    print(clust_df["size"].describe())

    print(">> canonicalization")
    # canonicalization based on entity_id from df
    from canonicalize import canonicalize_cluster, canonicalize_all  # use your functions
    canon_rules = {
        "Name_norm":  longest,
        "Street_norm": majority,
        "City_norm":   majority,
        "Zip_norm":    majority,
        "Email_norm":  most_frequent_valid,
        "Phone_norm":  most_frequent_valid,
    }
    # canonicalize_all(df, clusters, rules) in your notebook accepted an explicit list of cluster indices
    clusters = build_clusters(pred_pairs, df.index)
    df_eid, entities = canonicalize_all(df, clusters, canon_rules)
    df_eid = df_eid.drop(columns=['uid'], errors='ignore')

    print(">> save outputs")
    df_eid.to_csv(ROWS_WITH_EID_PATH, index=False)
    entities.to_csv(ENTITIES_PATH, index=False)

    print("done.")
    print(f"  pairs_pred -> {PAIRS_PRED_PATH}")
    print(f"  rows_with_entity_id -> {ROWS_WITH_EID_PATH}")
    print(f"  entities -> {ENTITIES_PATH}")


if __name__ == "__main__":
    main()

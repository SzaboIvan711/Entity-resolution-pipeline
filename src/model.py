# src/model.py
import joblib, numpy as np
from typing import Iterable, Tuple, Set
from rules import pair_features, is_match

def load_pair_model(path: str):
    blob = joblib.load(path)
    return blob['clf'], blob['feat_cols'], blob['threshold']

def model_score_pair(df, i, j, clf, feat_cols):
    f = pair_features(df, i, j)
    x = np.array([[f.get(c,0) for c in feat_cols]])
    return float(clf.predict_proba(x)[0,1])

def hybrid_predict_pairs(df, cand_pairs: Iterable[Tuple[int,int]], clf, feat_cols, thr) -> Set[Tuple[int,int]]:
    out=set()
    for i,j in cand_pairs:
        if is_match(df, i, j) or model_score_pair(df, i, j, clf, feat_cols) >= thr:
            out.add((i,j))
    return out

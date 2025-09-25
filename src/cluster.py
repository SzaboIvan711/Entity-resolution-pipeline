# cluster.py
from collections import defaultdict, deque
from itertools import combinations
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz import fuzz

def build_clusters(pairs, index):
    adj = defaultdict(set)
    for i,j in pairs:
        adj[i].add(j); adj[j].add(i)
    visited=set(); clusters=[]
    for v in index:
        if v in visited: continue
        comp=[]; q=deque([v]); visited.add(v)
        while q:
            u=q.popleft(); comp.append(u)
            for w in adj[u]:
                if w not in visited:
                    visited.add(w); q.append(w)
        clusters.append(sorted(comp))
    return clusters

# cluster.py

def summarize_clusters(df, clusters, uid_col='uid'):
    rows = []
    has_uid = uid_col in df.columns
    for cid, idxs in enumerate(clusters):
        size = len(idxs)
        n_uids = top_uid = top_uid_share = None
        if has_uid and size > 0:
            vc = pd.Series(df.loc[idxs, uid_col]).value_counts()
            n_uids = int(vc.size)
            if len(vc):
                top_uid = vc.index[0]
                top_uid_share = float(vc.iloc[0]) / size
        rows.append({
            'cluster_id': cid,
            'size': size,
            'n_uids': n_uids,
            'top_uid': top_uid,
            'top_uid_share': top_uid_share,
        })
    return pd.DataFrame(rows).sort_values(['size'], ascending=False)

def show_cluster(df, clusters, cid, cols=None, uid_col='uid'):
    idxs = clusters[cid]
    sub = df.loc[idxs].copy()
    # сортировку по uid делаем только если он есть
    if uid_col in sub.columns:
        sub = sub.sort_values(uid_col)
    return sub[cols] if cols else sub

def cluster_cohesion(df, idxs):
    name_min, street_min = 1.0, 100.0
    for i,j in combinations(idxs,2):
        a,b = df.loc[i], df.loc[j]
        name_min = min(name_min, JaroWinkler.normalized_similarity(a['Name_norm'], b['Name_norm']))
        street_min = min(street_min, fuzz.token_set_ratio(a['Street_norm'], b['Street_norm']))
    return {'name_min': name_min, 'street_min': street_min}

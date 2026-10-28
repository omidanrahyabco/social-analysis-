"""Descriptive temporal network analysis: snapshots, metrics, groups, balance, roles."""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np

from .io import Cohort

STRONG = 3.0      # mean friendship (both directions) that counts as a friendship tie
WEAK = 2.0        # mean friendship that still counts for group detection
HOSTILE_MEAN = 2.0
HOSTILE_MAX = 3.0


# --------------------------------------------------------------------------- tie states
def sym(M):
    return (M + M.T) / 2


def signed(F, H):
    """+1 friendship, -1 hostility, 0 neutral, for one wave (symmetric)."""
    Fm, Hm, Hx = sym(F), sym(H), np.maximum(H, H.T)
    neg = (Hm >= HOSTILE_MEAN) | (Hx >= HOSTILE_MAX)
    S = np.where(neg, -1, np.where(Fm >= STRONG, 1, 0))
    np.fill_diagonal(S, 0)
    return S


def romantic_state(Rr):
    """Couples, exes and one-way crushes for one wave of the romantic layer."""
    n = len(Rr)
    couples, exes, crushes = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if i < j and Rr[i, j] == 2 and Rr[j, i] == 2:
                couples.append((i, j))
            elif i < j and Rr[i, j] == 3 and Rr[j, i] == 3:
                exes.append((i, j))
            if Rr[i, j] == 1:
                crushes.append((i, j, bool(Rr[j, i] == 1)))
    return couples, exes, crushes


def positive_graph(F, S, ids, threshold=STRONG):
    Fm = sym(F)
    G = nx.Graph()
    G.add_nodes_from(range(len(ids)))
    for i, j in itertools.combinations(range(len(ids)), 2):
        if S[i, j] >= 0 and Fm[i, j] >= threshold:
            G.add_edge(i, j, weight=float(Fm[i, j] - threshold + 1), dist=1.0 / Fm[i, j])
    return G


# --------------------------------------------------------------------------- groups
def detect_groups(F, S, ids, seeds=40):
    G = positive_graph(F, S, ids, threshold=WEAK)
    best, best_q = None, -1
    for seed in range(seeds):
        comms = nx.community.louvain_communities(G, weight="weight", seed=seed, resolution=1.0)
        q = nx.community.modularity(G, comms, weight="weight")
        if q > best_q + 1e-9:
            best, best_q = comms, q
    best = sorted((sorted(c) for c in best), key=lambda c: (-len(c), c))
    return best, best_q


def track_groups(partitions, min_jaccard=0.34):
    """Give groups stable labels across waves and name what happened to them."""
    labels, history, counter = [], [], 0
    prev = None
    for t, comms in enumerate(partitions):
        cur_labels = [None] * len(comms)
        if prev is not None:
            pairs = []
            for a, pc in enumerate(prev):
                for b, cc in enumerate(comms):
                    inter = len(set(pc) & set(cc))
                    jac = inter / len(set(pc) | set(cc))
                    if jac >= min_jaccard:
                        pairs.append((jac, inter, a, b))
            used_prev = set()
            for jac, inter, a, b in sorted(pairs, reverse=True):
                if a in used_prev or cur_labels[b] is not None:
                    continue
                cur_labels[b] = labels[-1][a]
                used_prev.add(a)
        for b in range(len(comms)):
            if cur_labels[b] is None:
                counter += 1
                cur_labels[b] = f"G{counter}"
        events = []
        if prev is not None:
            for a, pc in enumerate(prev):
                lbl = labels[-1][a]
                succ = [cur_labels[b] for b, cc in enumerate(comms) if len(set(pc) & set(cc)) >= 2]
                if len(succ) >= 2:
                    events.append(dict(kind="split", group=lbl, into=succ))
                elif lbl not in cur_labels and len(pc) >= 2:
                    events.append(dict(kind="dissolved", group=lbl, members=list(pc)))
            for b, cc in enumerate(comms):
                lbl = cur_labels[b]
                src = [labels[-1][a] for a, pc in enumerate(prev) if len(set(pc) & set(cc)) >= 2]
                if lbl in labels[-1]:
                    old = set(prev[labels[-1].index(lbl)])
                    joined, left = sorted(set(cc) - old), sorted(old - set(cc))
                    if joined or left:
                        events.append(dict(kind="changed", group=lbl, joined=joined, left=left))
                    else:
                        events.append(dict(kind="stable", group=lbl))
                elif len(cc) >= 2:
                    events.append(dict(kind="formed", group=lbl, from_=src))
        else:
            events = [dict(kind="formed", group=l, from_=[]) for l, c in zip(cur_labels, comms) if len(c) >= 2]
        labels.append(cur_labels)
        history.append(events)
        prev = comms
    return labels, history


# --------------------------------------------------------------------------- balance
def balance(S):
    n = len(S)
    counts = {"+++": 0, "+--": 0, "++-": 0, "---": 0}
    unbalanced = []
    middle = np.zeros(n, int)
    involved = np.zeros(n, int)
    for i, j, k in itertools.combinations(range(n), 3):
        s = [S[i, j], S[i, k], S[j, k]]
        if 0 in s:
            continue
        pos = sum(1 for x in s if x > 0)
        key = {3: "+++", 1: "+--", 2: "++-", 0: "---"}[pos]
        counts[key] += 1
        if key == "++-":
            # the person holding both positive ties is caught in the middle
            for m, a, b in ((i, j, k), (j, i, k), (k, i, j)):
                if S[m, a] > 0 and S[m, b] > 0:
                    middle[m] += 1
                    unbalanced.append(dict(middle=m, sides=[a, b]))
            for p in (i, j, k):
                involved[p] += 1
    total = sum(counts.values())
    tension = counts["++-"] / total if total else 0.0
    return dict(counts=counts, total=total, tension=tension, unbalanced=unbalanced,
                middle=middle, involved=involved)


# --------------------------------------------------------------------------- homophily
def ei_index(S, attr):
    """Krackhardt E-I index on positive ties: -1 all within-group, +1 all between."""
    n = len(S)
    e = i_ = 0
    for a, b in itertools.combinations(range(n), 2):
        if S[a, b] > 0:
            if attr[a] == attr[b]:
                i_ += 1
            else:
                e += 1
    return (e - i_) / (e + i_) if (e + i_) else 0.0


def ei_expected(attr):
    """E-I index if ties were random: the baseline to compare against."""
    n = len(attr)
    e = sum(1 for a, b in itertools.combinations(range(n), 2) if attr[a] != attr[b])
    tot = n * (n - 1) / 2
    return (e - (tot - e)) / tot


# --------------------------------------------------------------------------- per wave
def analyse_wave(c: Cohort, t: int):
    F, H, Rr = c.R["friendship"][t], c.R["hostility"][t], c.R["romantic"][t]
    n = c.n
    S = signed(F, H)
    D = (F >= STRONG).astype(int)
    np.fill_diagonal(D, 0)
    G = positive_graph(F, S, c.ids)
    wave = c.wave_ids[t]
    pos_pairs = {(i, j) for i, j in G.edges}
    shared = c.shared_foci[t]
    bal = balance(S)
    btw = nx.betweenness_centrality(G, weight="dist", normalized=True)
    gender = c.attr("gender")
    housing = c.attr("housing", wave)
    origin = np.where(c.attr("origin") == "Tehran", "Tehran", "other city")
    couples, exes, crushes = romantic_state(Rr)
    Fm = sym(F)
    metrics = dict(
        density=float(D.sum() / (n * (n - 1))),
        reciprocity=float((D * D.T).sum() / D.sum()) if D.sum() else 0.0,
        transitivity=float(nx.transitivity(G)),
        mean_degree=float(np.mean([d for _, d in G.degree])),
        positive_ties=int(len(pos_pairs)),
        negative_ties=int(((S < 0).sum()) // 2),
        tension=float(bal["tension"]),
        signed_triads=int(bal["total"]),
        ei_gender=float(ei_index(S, gender)),
        ei_housing=float(ei_index(S, housing)),
        ei_origin=float(ei_index(S, origin)),
        focus_share=float(np.mean([shared[i, j] > 0 for i, j in pos_pairs])) if pos_pairs else 0.0,
        asymmetry=float(np.abs(F - F.T)[~np.eye(n, dtype=bool)].mean()),
        mean_friendship=float(Fm[~np.eye(n, dtype=bool)].mean()),
        couples=len(couples),
    )
    baselines = dict(ei_gender=ei_expected(gender), ei_housing=ei_expected(housing),
                     ei_origin=ei_expected(origin))
    per_person = []
    for p in range(n):
        neg = int((S[p] < 0).sum())
        per_person.append(dict(
            degree=int(G.degree[p]), strength=float(sum(d["weight"] for _, _, d in G.edges(p, data=True))),
            betweenness=float(btw[p]), negative=neg, middle=int(bal["middle"][p]),
            gives=float(np.delete(F[p], p).mean()), receives=float(np.delete(F[:, p], p).mean()),
        ))
    return dict(S=S, G=G, metrics=metrics, baselines=baselines, balance=bal, people=per_person,
                couples=couples, exes=exes, crushes=crushes, positive=pos_pairs)


def roles(people, labels_t, comms_t):
    deg = np.array([p["degree"] for p in people])
    btw = np.array([p["betweenness"] for p in people])
    hub_cut = max(3, deg.mean() + deg.std())
    btw_rank = np.argsort(-btw)
    out = []
    for k, p in enumerate(people):
        r = []
        if p["degree"] >= hub_cut:
            r.append("hub")
        if k in btw_rank[:2] and p["betweenness"] > 0.15:
            r.append("broker")
        if p["middle"] > 0:
            r.append("caught in the middle")
        if p["negative"] >= 2:
            r.append("in open conflict")
        elif p["negative"] == 1:
            r.append("one enemy")
        if p["degree"] <= 1:
            r.append("peripheral")
        if not r:
            r.append("group member")
        out.append(r)
    return out


def turnover(prev_pairs, cur_pairs):
    union = prev_pairs | cur_pairs
    return len(prev_pairs & cur_pairs) / len(union) if union else 1.0


def analyse(c: Cohort):
    waves = [analyse_wave(c, t) for t in range(c.T)]
    partitions = []
    for t, w in enumerate(waves):
        comms, q = detect_groups(c.R["friendship"][t], w["S"], c.ids)
        w["modularity"] = q
        partitions.append(comms)
    labels, history = track_groups(partitions)
    for t, w in enumerate(waves):
        w["groups"] = partitions[t]
        w["group_labels"] = labels[t]
        w["group_events"] = history[t]
        w["roles"] = roles(w["people"], labels[t], partitions[t])
        w["metrics"]["groups"] = int(sum(1 for g in partitions[t] if len(g) >= 2))
        if t:
            prev = waves[t - 1]["positive"]
            w["metrics"]["stability"] = float(turnover(prev, w["positive"]))
            new = w["positive"] - prev
            shared = c.shared_foci[t]
            w["metrics"]["new_ties"] = len(new)
            w["metrics"]["lost_ties"] = len(prev - w["positive"])
            w["metrics"]["new_tie_focus_share"] = (float(np.mean([shared[i, j] > 0 for i, j in new]))
                                                   if new else None)
        else:
            w["metrics"]["stability"] = None
            w["metrics"]["new_ties"] = len(w["positive"])
            w["metrics"]["lost_ties"] = 0
            w["metrics"]["new_tie_focus_share"] = w["metrics"]["focus_share"]
    return waves

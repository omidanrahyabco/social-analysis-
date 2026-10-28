"""Explanatory model: what drives changes in friendship between waves.

A tie-change regression. For every directed pair (i -> j) and every pair of
consecutive waves, the outcome is the change in i's friendship rating of j,
explained by the state at the earlier wave and the contexts at the later one.
This is a lightweight stand-in for a stochastic actor-oriented model (RSiena),
which is the method to use on the full 30-person, 6-semester data.
"""
from __future__ import annotations

import itertools

import numpy as np

from .io import Cohort
from .network import STRONG, balance, signed, sym

NEGATIVE_EVENTS = {"breakup", "fight", "public_fight", "cold_shoulder", "crush_rejected",
                   "gossip", "group_dissolved", "betrayal"}
POSITIVE_EVENTS = {"started_dating", "group_formed", "support", "moved_housing", "reconciliation"}

FEATURES = [
    ("level", "Current friendship level", "state dependence: high ties have less room to grow"),
    ("recip_gap", "The other person rates the tie higher", "reciprocity: people adjust toward the other side"),
    ("common", "Common friends", "triadic closure: friends of friends become friends"),
    ("shared", "Shared contexts next wave", "propinquity: dorm, class section, study group"),
    ("shared_late", "Shared contexts x semester 2", "do contexts matter less once everyone knows everyone"),
    ("lost_context", "Lost a shared context", "moving out, changing section"),
    ("same_gender", "Same gender", "homophily"),
    ("same_housing", "Same housing type", "homophily: dorm, family home, rented"),
    ("gpa_gap", "GPA gap (points out of 20)", "academic homophily"),
    ("attract_j", "Attractiveness of the other person", "status, attraction"),
    ("hostility", "Hostility between them", "conflict erodes friendship"),
    ("unbalanced", "Unbalanced triangles on this tie", "balance pressure: Heider"),
    ("neg_event", "Negative events between them", "breakups, fights, rejection (intensity)"),
    ("pos_event", "Positive events between them", "dating, joining a group, support (intensity)"),
]


def event_matrices(c: Cohort):
    """Summed intensity of positive and negative events per pair between waves t and t+1."""
    n, T = c.n, c.T
    idx = {p: k for k, p in enumerate(c.ids)}
    neg = np.zeros((T, n, n))
    pos = np.zeros((T, n, n))
    dates = c.waves["date"].tolist()
    for _, e in c.events.iterrows():
        t = next((k for k in range(1, T) if dates[k - 1] < e.date <= dates[k]), None)
        if t is None:
            continue
        actors = [idx[a] for a in e.actors.split(";") if a in idx]
        target = neg if e.type in NEGATIVE_EVENTS else pos if e.type in POSITIVE_EVENTS else None
        if target is None:
            continue
        for a, b in itertools.permutations(actors, 2):
            target[t, a, b] += e.intensity
    return neg, pos


def features(F, H, F_other_wave_shared_now, shared_next, late, same_gender, same_housing,
             gpa, attract, neg_ev, pos_ev):
    """Feature tensor (n, n, k) for one transition. Shared by fitting and simulation."""
    n = len(F)
    Fm = sym(F)
    S = signed(F, H)
    strong = (Fm >= STRONG) & (S >= 0)
    np.fill_diagonal(strong, False)
    common = strong.astype(float) @ strong.astype(float)
    # unbalanced (++-) triangles that pass through each tie
    unb = np.zeros((n, n))
    P, N = (S > 0).astype(float), (S < 0).astype(float)
    unb += (P @ P) * N           # tie is the negative side of a ++- triangle
    unb += (P @ N) * P + (N @ P) * P   # tie is a positive side
    X = np.stack([
        F,
        F.T - F,
        common,
        shared_next,
        shared_next * late,
        np.maximum(F_other_wave_shared_now - shared_next, 0),
        same_gender,
        same_housing,
        np.abs(gpa[:, None] - gpa[None, :]),
        np.broadcast_to(attract[None, :], (n, n)),
        sym(H),
        unb,
        neg_ev,
        pos_ev,
    ], axis=-1)
    return X


def person_arrays(c: Cohort, t: int):
    wave = c.wave_ids[t]
    g = c.attr("gender")
    h = c.attr("housing", wave)
    return dict(same_gender=(g[:, None] == g[None, :]).astype(float),
                same_housing=(h[:, None] == h[None, :]).astype(float),
                gpa=c.attr("gpa", wave).astype(float),
                attract=c.attr("attractiveness_peer").astype(float))


def build_design(c: Cohort):
    neg, pos = event_matrices(c)
    sem = c.waves["semester"].tolist()
    rows_X, rows_y, rows_pair, rows_t = [], [], [], []
    off = ~np.eye(c.n, dtype=bool)
    for t in range(c.T - 1):
        F, H = c.R["friendship"][t], c.R["hostility"][t]
        pa = person_arrays(c, t + 1)
        late = float(sem[t + 1] != sem[0])
        X = features(F, H, c.shared_foci[t], c.shared_foci[t + 1], late, pa["same_gender"],
                     pa["same_housing"], pa["gpa"], pa["attract"], neg[t + 1], pos[t + 1])
        y = c.R["friendship"][t + 1] - F
        ii, jj = np.where(off)
        rows_X.append(X[ii, jj])
        rows_y.append(y[ii, jj])
        rows_pair.append(np.minimum(ii, jj) * c.n + np.maximum(ii, jj))
        rows_t.append(np.full(len(ii), t))
    return (np.vstack(rows_X), np.concatenate(rows_y), np.concatenate(rows_pair),
            np.concatenate(rows_t))


def ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return beta, resid


def fit(c: Cohort, n_boot=2000, seed=7):
    X, y, pair, _ = build_design(c)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    beta_z, resid = ols(Z, y)
    beta_raw, _ = ols(X, y)
    r2 = 1 - resid.var() / y.var()
    # cluster bootstrap over unordered pairs (both directions and all waves move together)
    rng = np.random.default_rng(seed)
    clusters = np.unique(pair)
    members = {p: np.where(pair == p)[0] for p in clusters}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(clusters, len(clusters), replace=True)
        rows = np.concatenate([members[p] for p in pick])
        b, _ = ols(Z[rows], y[rows])
        boots.append(b)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [5, 95], axis=0)
    coefs = []
    for k, (key, label, theory) in enumerate(FEATURES):
        coefs.append(dict(key=key, label=label, theory=theory, beta=float(beta_z[k + 1]),
                          lo=float(lo[k + 1]), hi=float(hi[k + 1]), raw=float(beta_raw[k + 1]),
                          sd=float(sd[k]),
                          clear=bool(lo[k + 1] > 0 or hi[k + 1] < 0)))
    return dict(coefs=coefs, intercept_raw=float(beta_raw[0]), r2=float(r2),
                sigma=float(resid.std()), n_obs=int(len(y)), n_pairs=int(len(clusters)),
                beta_raw=beta_raw)


def romance_summary(c: Cohort):
    """Relationship spells from the events log (start, end, months)."""
    spells, open_ = [], {}
    for _, e in c.events.iterrows():
        pair_ = tuple(sorted(e.actors.split(";")))
        if e.type == "started_dating":
            open_[pair_] = e.date
        elif e.type == "breakup" and pair_ in open_:
            start = open_.pop(pair_)
            spells.append(dict(pair=list(pair_), start=str(start.date()), end=str(e.date.date()),
                               months=round((e.date - start).days / 30.4, 1), ended=True))
    last = c.waves["date"].iloc[-1]
    for pair_, start in open_.items():
        spells.append(dict(pair=list(pair_), start=str(start.date()), end=None,
                           months=round((last - start).days / 30.4, 1), ended=False))
    exposure = sum(s["months"] for s in spells)
    ended = sum(s["ended"] for s in spells)
    return dict(spells=spells, exposure_months=exposure, breakups=ended,
                monthly_hazard=(ended / exposure) if exposure else None)

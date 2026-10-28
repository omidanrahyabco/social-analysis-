"""Agent-based simulation: replay the observed period (back-test) and forecast ahead.

Friendship changes come from the fitted tie-change model (models.fit). Romance,
conflict and side-taking follow explicit rules whose rates are listed in
PARAMS. With only one observed breakup, those rates are mostly priors; the full
dataset is what should pin them down.
"""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np

from .io import Cohort
from .models import features, person_arrays, romance_summary
from .network import STRONG, WEAK, signed, sym

PARAMS = dict(
    months_per_wave=2.5,
    breakup_prior=(1.0, 12.0),       # Gamma prior: about one breakup per couple-year
    crush_reciprocate=0.10,
    crush_fade=0.30,
    taken_crush=0.5,                 # interest outside a relationship, relative to singles
    wandering_breakup=2.0,
    hostility_decay=0.85,
    side_taking=0.40,
    jealousy=0.60,
    rivalry=0.35,
    fight=0.15,
    balance_resolution=0.30,
)


def romance_rates(c: Cohort, prior_onset=(1, 30), prior_couple=(2, 4)):
    """Per-wave rates estimated from the romantic layer, shrunk toward a Beta prior.

    onset:  a directed opposite-gender pair with no interest starts one (crush or dating)
    couple: a mutual crush between two single people becomes a relationship
    """
    Rr, g = c.R["romantic"], c.attr("gender")
    onsets = at_risk = formed = mutual = 0
    for t in range(c.T - 1):
        for i in range(c.n):
            for j in range(c.n):
                if i == j or g[i] == g[j]:
                    continue
                if Rr[t, i, j] == 0 and Rr[t, j, i] != 3:
                    at_risk += 1
                    onsets += Rr[t + 1, i, j] in (1, 2)
                if i < j and Rr[t, i, j] == 1 and Rr[t, j, i] == 1:
                    mutual += 1
                    formed += Rr[t + 1, i, j] == 2 and Rr[t + 1, j, i] == 2
    a, b = prior_onset
    a2, b2 = prior_couple
    return dict(onsets=int(onsets), at_risk=int(at_risk), onset=(a + onsets) / (a + b + at_risk),
                mutual=int(mutual), formed=int(formed), couple=(a2 + formed) / (a2 + b2 + mutual))


def breakup_probability(c: Cohort, p=PARAMS):
    rs = romance_summary(c)
    a, b = p["breakup_prior"]
    hazard = (a + rs["breakups"]) / (b + rs["exposure_months"])
    return 1 - np.exp(-hazard * p["months_per_wave"]), hazard


class Model:
    def __init__(self, c: Cohort, fitted, p=PARAMS):
        self.c, self.p = c, p
        self.beta = fitted["beta_raw"]
        self.sigma = fitted["sigma"]
        self.p_break, self.hazard = breakup_probability(c, p)
        self.rates = romance_rates(c)
        self.gender = c.attr("gender")
        self.opposite = self.gender[:, None] != self.gender[None, :]
        self.E = c.attr("extraversion").astype(float)
        self.attract = c.attr("attractiveness_peer").astype(float)

    # ------------------------------------------------------------------ one wave
    def step(self, F, H, Rr, shared_now, shared_next, late, pa, rng):
        p, n = self.p, len(F)
        F, H, Rr = F.copy(), H.copy(), Rr.copy()
        neg = np.zeros((n, n))
        pos = np.zeros((n, n))
        Fm = sym(F)
        log = []

        def couple_of(i):
            js = [j for j in range(n) if Rr[i, j] == 2 and Rr[j, i] == 2]
            return js[0] if js else None

        # breakups: twice as likely while either partner is interested in someone else
        for i, j in itertools.combinations(range(n), 2):
            if not (Rr[i, j] == 2 and Rr[j, i] == 2):
                continue
            wandering = (Rr[i] == 1).any() or (Rr[j] == 1).any()
            if rng.random() < self.p_break * (p["wandering_breakup"] if wandering else 1):
                Rr[i, j] = Rr[j, i] = 3
                neg[i, j] += 5
                neg[j, i] += 5
                hurt, other = (i, j) if rng.random() < 0.5 else (j, i)
                H[hurt, other] += 4
                H[other, hurt] += 2
                for k in range(n):
                    if k in (i, j):
                        continue
                    for a, b in ((i, j), (j, i)):
                        if F[k, a] >= 4 and F[k, a] > F[k, b] + 1 and rng.random() < p["side_taking"]:
                            H[k, b] += 1.5
                            neg[k, b] += 1
                log.append(("breakup", i, j))
        # crushes become couples, one-way crushes are returned or fade
        for i, j in itertools.combinations(range(n), 2):
            if Rr[i, j] == 1 and Rr[j, i] == 1 and couple_of(i) is None and couple_of(j) is None:
                pr = self.rates["couple"] + 0.06 * (self.E[i] + self.E[j] - 6) + 0.05 * (Fm[i, j] - 3)
                if rng.random() < np.clip(pr, 0.05, 0.7):
                    Rr[i, j] = Rr[j, i] = 2
                    pos[i, j] += 3
                    pos[j, i] += 3
                    for a, b in ((i, j), (j, i)):
                        for x in range(n):
                            if x not in (i, j) and Rr[x, a] == 3 and rng.random() < p["jealousy"]:
                                H[x, b] += 3
                                H[b, x] += 2
                                neg[x, b] += 2
                                neg[b, x] += 2
                            if x not in (i, j) and Rr[x, a] == 1 and rng.random() < p["rivalry"]:
                                H[x, b] += 0.8
                    log.append(("couple", i, j))
        for i in range(n):
            for j in range(n):
                if i == j or Rr[i, j] != 1 or Rr[j, i] != 0:
                    continue
                partner = couple_of(j)
                if partner is None:
                    pr = p["crush_reciprocate"] + 0.07 * (self.attract[i] - 3) + 0.05 * (F[j, i] - 2)
                    if rng.random() < np.clip(pr, 0.02, 0.4):
                        Rr[j, i] = 1
                        continue
                elif rng.random() < p["rivalry"]:
                    H[i, partner] += 0.7
                    H[partner, i] += 0.7
                if rng.random() < p["crush_fade"] + (0.1 if partner is not None else 0):
                    Rr[i, j] = 0
        # new crushes (people in a couple too, less often)
        for i in range(n):
            if (Rr[i] == 1).any():
                continue
            own = p["taken_crush"] if couple_of(i) is not None else 1.0
            for j in range(n):
                if i == j or not self.opposite[i, j] or Rr[i, j] == 3 or Rr[j, i] == 3:
                    continue
                meets = 1.0 if (Fm[i, j] >= 2.5 or shared_next[i, j] > 0) else 0.3
                taken = 0.4 if couple_of(j) is not None else 1.0
                if rng.random() < self.rates["onset"] * self.attract[j] / self.attract.mean() * meets * taken * own:
                    Rr[i, j] = 1
                    break
        # escalation between people who already dislike each other and still meet
        Hm = sym(H)
        for i, j in itertools.combinations(range(n), 2):
            if Hm[i, j] >= 2 and (shared_next[i, j] > 0 or rng.random() < 0.3) and rng.random() < p["fight"]:
                neg[i, j] += 3
                neg[j, i] += 3
                H[i, j] += 1
                H[j, i] += 1
                log.append(("fight", i, j))

        # friendship update from the fitted model
        X = features(F, H, shared_now, shared_next, late, pa["same_gender"], pa["same_housing"],
                     pa["gpa"], pa["attract"], neg, pos)
        mean = self.beta[0] + X @ self.beta[1:]
        shared_noise = rng.normal(0, 1, (n, n))
        shared_noise = (shared_noise + shared_noise.T) / np.sqrt(2)
        noise = self.sigma * (0.6 * shared_noise + 0.8 * rng.normal(0, 1, (n, n)))
        F = np.clip(F + mean + noise, 0, 5)
        np.fill_diagonal(F, 0)

        # hostility fades unless fed
        H = np.clip(H * p["hostility_decay"], 0, 5)
        np.fill_diagonal(H, 0)

        # balance pressure: the person in the middle of a ++- triangle drifts from one side
        S = signed(F, H)
        for m in range(n):
            for a, b in itertools.combinations(range(n), 2):
                if m in (a, b) or not (S[m, a] > 0 and S[m, b] > 0 and S[a, b] < 0):
                    continue
                if rng.random() < p["balance_resolution"]:
                    weak = a if F[m, a] < F[m, b] else b
                    F[m, weak] = max(F[m, weak] - 1, 0)
                    F[weak, m] = max(F[weak, m] - 0.5, 0)
        return F, H, Rr, log


# ---------------------------------------------------------------------- metrics
def quick_metrics(F, H, Rr, gender):
    n = len(F)
    S = signed(F, H)
    A = (S > 0).astype(float)
    D = (F >= STRONG).astype(float)
    np.fill_diagonal(D, 0)
    deg = A.sum(1)
    tri = np.trace(A @ A @ A) / 6
    triples = (deg * (deg - 1) / 2).sum()
    total = unb = 0
    for i, j, k in itertools.combinations(range(n), 3):
        s = (S[i, j], S[i, k], S[j, k])
        if 0 in s:
            continue
        total += 1
        unb += sum(1 for x in s if x > 0) == 2
    e = i_ = 0
    for a, b in itertools.combinations(range(n), 2):
        if S[a, b] > 0:
            if gender[a] == gender[b]:
                i_ += 1
            else:
                e += 1
    couples = sum(1 for a, b in itertools.combinations(range(n), 2) if Rr[a, b] == 2 and Rr[b, a] == 2)
    off = ~np.eye(n, dtype=bool)
    return dict(
        density=float(D.sum() / (n * (n - 1))),
        positive_ties=int(A.sum() // 2),
        negative_ties=int((S < 0).sum() // 2),
        transitivity=float(3 * tri / triples) if triples else 0.0,
        tension=float(unb / total) if total else 0.0,
        mean_friendship=float(sym(F)[off].mean()),
        ei_gender=float((e - i_) / (e + i_)) if e + i_ else 0.0,
        couples=couples,
    )


METRIC_KEYS = ["density", "positive_ties", "negative_ties", "transitivity", "tension",
               "mean_friendship", "ei_gender", "couples"]


def summarise(samples):
    a = np.array(samples, dtype=float)
    return dict(median=float(np.median(a)), lo=float(np.percentile(a, 10)), hi=float(np.percentile(a, 90)))


# ---------------------------------------------------------------------- back-test
def backtest(c: Cohort, fitted, runs=500, seed=11):
    """Start from the first wave, replay contexts as observed, let the model make the drama."""
    model = Model(c, fitted)
    rng = np.random.default_rng(seed)
    sem = c.waves["semester"].tolist()
    gender = c.attr("gender")
    traj = {k: [[] for _ in range(c.T)] for k in METRIC_KEYS}
    for _ in range(runs):
        F = c.R["friendship"][0].copy()
        H = c.R["hostility"][0].copy()
        Rr = c.R["romantic"][0].copy()
        for k, v in quick_metrics(F, H, Rr, gender).items():
            traj[k][0].append(v)
        for t in range(c.T - 1):
            pa = person_arrays(c, t + 1)
            F, H, Rr, _ = model.step(F, H, Rr, c.shared_foci[t], c.shared_foci[t + 1],
                                     float(sem[t + 1] != sem[0]), pa, rng)
            for k, v in quick_metrics(F, H, Rr, gender).items():
                traj[k][t + 1].append(v)
    observed = [quick_metrics(c.R["friendship"][t], c.R["hostility"][t], c.R["romantic"][t], gender)
                for t in range(c.T)]
    out = {}
    for k in METRIC_KEYS:
        bands = [summarise(traj[k][t]) for t in range(c.T)]
        obs = [o[k] for o in observed]
        last = bands[-1]
        out[k] = dict(bands=bands, observed=obs,
                      inside=bool(last["lo"] - 1e-9 <= obs[-1] <= last["hi"] + 1e-9))
    return out


def one_step(c: Cohort, fitted, runs=300, seed=5):
    """Start from each observed wave, predict the next one, score it against what happened.

    Scores the probability of each friendship tie with the Brier score (lower is better)
    against a persistence baseline that assumes every tie stays as it is.
    """
    model = Model(c, fitted)
    rng = np.random.default_rng(seed)
    sem = c.waves["semester"].tolist()
    n = c.n
    iu = np.triu_indices(n, 1)
    out = []
    for t in range(c.T - 1):
        F, H, Rr = (c.R[k][t] for k in ("friendship", "hostility", "romantic"))
        pa = person_arrays(c, t + 1)
        p_pos = np.zeros((n, n))
        p_neg = np.zeros((n, n))
        for _ in range(runs):
            F1, H1, _, _ = model.step(F, H, Rr, c.shared_foci[t], c.shared_foci[t + 1],
                                      float(sem[t + 1] != sem[0]), pa, rng)
            S1 = signed(F1, H1)
            p_pos += S1 > 0
            p_neg += S1 < 0
        p_pos /= runs
        p_neg /= runs
        S_now = signed(F, H)
        S_next = signed(c.R["friendship"][t + 1], c.R["hostility"][t + 1])
        y = (S_next > 0)[iu].astype(float)
        y_neg = (S_next < 0)[iu].astype(float)
        brier = float(np.mean((p_pos[iu] - y) ** 2))
        brier_persist = float(np.mean(((S_now > 0)[iu] - y) ** 2))
        changed = (S_now > 0)[iu] != (S_next > 0)[iu]
        hits = float(np.mean(np.abs(p_pos[iu][changed] - y[changed]) < 0.5)) if changed.any() else None
        out.append(dict(from_wave=c.wave_ids[t], to_wave=c.wave_ids[t + 1], brier=brier,
                        brier_persistence=brier_persist, changed=int(changed.sum()),
                        changed_called=hits,
                        brier_negative=float(np.mean((p_neg[iu] - y_neg) ** 2)),
                        brier_negative_persistence=float(np.mean(((S_now < 0)[iu] - y_neg) ** 2))))
    return out


# ---------------------------------------------------------------------- forecast
def future_contexts(c: Cohort, rng, section_size=5):
    """Contexts for the next semester: homes and dorms stay, sections are reshuffled."""
    last = c.wave_ids[-1]
    idx = {p: k for k, p in enumerate(c.ids)}
    n = c.n
    shared = np.zeros((n, n))
    f = c.foci[c.foci.wave == last]
    keep_prob = {"dorm": 1.0, "rented": 1.0, "study": 0.75, "extracurricular": 0.7}
    for (focus, ftype), g in f.groupby(["focus", "type"]):
        members = [idx[p] for p in g.id]
        if len(members) >= n or ftype not in keep_prob:
            continue
        if rng.random() < keep_prob[ftype]:
            for a, b in itertools.permutations(members, 2):
                shared[a, b] += 1
    order = rng.permutation(n)
    for s in range(0, n, section_size):
        sec = order[s:s + section_size]
        for a, b in itertools.permutations(sec, 2):
            shared[a, b] += 1
    return shared


def forecast(c: Cohort, fitted, steps=2, runs=1000, seed=23):
    model = Model(c, fitted)
    rng = np.random.default_rng(seed)
    n, gender = c.n, c.attr("gender")
    pa = person_arrays(c, c.T - 1)
    F0, H0, R0 = (c.R[k][-1] for k in ("friendship", "hostility", "romantic"))
    S0 = signed(F0, H0)
    couples0 = [(i, j) for i, j in itertools.combinations(range(n), 2) if R0[i, j] == 2 and R0[j, i] == 2]
    triads0 = [(m, a, b) for m in range(n) for a, b in itertools.combinations(range(n), 2)
               if m not in (a, b) and S0[m, a] > 0 and S0[m, b] > 0 and S0[a, b] < 0]

    p_pos = np.zeros((n, n))
    p_neg = np.zeros((n, n))
    p_date = np.zeros((n, n))
    comember = np.zeros((n, n))
    survive = {cp: 0 for cp in couples0}
    triad_out = {t: dict(unresolved=0, drop_a=0, drop_b=0, reconciled=0, other=0) for t in triads0}
    deg_sum = np.zeros(n)
    isolated = np.zeros(n)
    conflict = np.zeros(n)
    traj = {k: [[] for _ in range(steps)] for k in METRIC_KEYS}
    fights = np.zeros((n, n))

    shared_now = c.shared_foci[-1]
    for _ in range(runs):
        F, H, Rr = F0.copy(), H0.copy(), R0.copy()
        ctx = future_contexts(c, rng)
        now = shared_now
        dated = np.zeros((n, n), bool)
        for s in range(steps):
            F, H, Rr, log = model.step(F, H, Rr, now, ctx, 1.0, pa, rng)
            now = ctx
            dated |= (Rr == 2) & (Rr.T == 2)
            for kind, i, j in log:
                if kind == "fight":
                    fights[i, j] += 1
                    fights[j, i] += 1
            for k, v in quick_metrics(F, H, Rr, gender).items():
                traj[k][s].append(v)
        S = signed(F, H)
        p_pos += S > 0
        p_neg += S < 0
        p_date += dated
        A = (S > 0)
        deg = A.sum(1)
        deg_sum += deg
        isolated += deg <= 1
        conflict += (S < 0).any(1)
        for cp in couples0:
            survive[cp] += Rr[cp] == 2 and Rr[cp[::-1]] == 2
        for (m, a, b) in triads0:
            o = triad_out[(m, a, b)]
            if S[a, b] >= 0:
                o["reconciled"] += 1
            elif S[m, a] > 0 and S[m, b] > 0:
                o["unresolved"] += 1
            elif S[m, a] > 0 and S[m, b] <= 0:
                o["drop_b"] += 1
            elif S[m, b] > 0 and S[m, a] <= 0:
                o["drop_a"] += 1
            else:
                o["other"] += 1
        G = nx.Graph()
        G.add_nodes_from(range(n))
        Fm = sym(F)
        for i, j in itertools.combinations(range(n), 2):
            if S[i, j] >= 0 and Fm[i, j] >= WEAK:
                G.add_edge(i, j, weight=Fm[i, j] - WEAK + 1)
        for comm in nx.community.louvain_communities(G, weight="weight", seed=int(rng.integers(1e9))):
            for a, b in itertools.permutations(comm, 2):
                comember[a, b] += 1

    # consensus groups from co-membership
    CG = nx.Graph()
    CG.add_nodes_from(range(n))
    cm = comember / runs
    for i, j in itertools.combinations(range(n), 2):
        if cm[i, j] >= 0.3:
            CG.add_edge(i, j, weight=cm[i, j])
    consensus = nx.community.louvain_communities(CG, weight="weight", seed=1)
    consensus = sorted((sorted(g) for g in consensus), key=lambda g: (-len(g), g))

    return dict(
        runs=runs, steps=steps,
        p_positive=(p_pos / runs).tolist(), p_negative=(p_neg / runs).tolist(),
        p_dating=(p_date / runs).tolist(), comembership=cm.tolist(), consensus=consensus,
        couple_survival={f"{i}-{j}": float(survive[(i, j)] / runs) for i, j in couples0},
        couples_now=couples0,
        triads=[dict(middle=m, a=a, b=b, **{k: v / runs for k, v in triad_out[(m, a, b)].items()})
                for (m, a, b) in triads0],
        expected_degree=(deg_sum / runs).tolist(), p_isolated=(isolated / runs).tolist(),
        p_conflict=(conflict / runs).tolist(), p_fight=(fights / runs).tolist(),
        trajectories={k: [summarise(traj[k][s]) for s in range(steps)] for k in METRIC_KEYS},
        breakup_prob_per_wave=float(model.p_break), breakup_hazard_month=float(model.hazard),
        romance_rates=model.rates, params=dict(PARAMS),
    )

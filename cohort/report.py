"""Assemble all results into one JSON document and an interactive HTML report."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import networkx as nx
import numpy as np

from .io import Cohort, LAYERS
from .models import NEGATIVE_EVENTS, POSITIVE_EVENTS, romance_summary
from .network import sym

TEMPLATE = Path(__file__).with_name("report_template.html")


def _pct(p):
    return f"{round(100 * p)}%"


def layout(c: Cohort, seed=3):
    """One fixed position per person, so movement between waves means change, not redrawing."""
    G = nx.Graph()
    G.add_nodes_from(range(c.n))
    Fm = np.mean([sym(c.R["friendship"][t]) for t in range(c.T)], axis=0)
    for i in range(c.n):
        for j in range(i + 1, c.n):
            if Fm[i, j] >= 1.5:
                G.add_edge(i, j, weight=float(Fm[i, j]) ** 2)
    pos = nx.kamada_kawai_layout(G, weight=None) if nx.is_connected(G) else nx.spring_layout(G, seed=seed)
    pos = nx.spring_layout(G, pos=pos, weight="weight", seed=seed, iterations=60, k=0.9)
    xy = np.array([pos[i] for i in range(c.n)])
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    return {c.ids[i]: [round(float(xy[i, 0]), 4), round(float(xy[i, 1]), 4)] for i in range(c.n)}


def build(c: Cohort, waves, fitted, one_step, free_run, fc, dataset_name):
    ids = c.ids
    name = lambda k: ids[k]
    T, n = c.T, c.n

    # ------------------------------------------------------------------ waves
    W = []
    for t, w in enumerate(waves):
        ev = []
        for e in w["group_events"]:
            e = dict(e)
            for key in ("joined", "left", "members"):
                if key in e:
                    e[key] = [name(k) for k in e[key]]
            if "from_" in e:
                e["from"] = e.pop("from_")
            ev.append(e)
        W.append(dict(
            metrics=w["metrics"], baselines=w["baselines"], modularity=w["modularity"],
            groups=[dict(label=l, members=[name(k) for k in g]) for l, g in zip(w["group_labels"], w["groups"])],
            group_events=ev,
            unbalanced=[dict(middle=name(u["middle"]), sides=[name(s) for s in u["sides"]])
                        for u in w["balance"]["unbalanced"]],
            triads=w["balance"]["counts"],
            roles={name(k): r for k, r in enumerate(w["roles"])},
            people={name(k): p for k, p in enumerate(w["people"])},
            couples=[[name(a), name(b)] for a, b in w["couples"]],
            exes=[[name(a), name(b)] for a, b in w["exes"]],
            crushes=[[name(a), name(b), m] for a, b, m in w["crushes"]],
            S=w["S"].astype(int).tolist(),
        ))

    # ------------------------------------------------------------------ inputs
    persons = c.persons.to_dict("records")
    pw = {}
    for r in c.person_wave.to_dict("records"):
        pw.setdefault(r["id"], {})[r["wave"]] = dict(housing=r["housing"], gpa=r["gpa"])
    foci = []
    for (focus, ftype), g in c.foci.groupby(["focus", "type"], sort=False):
        foci.append(dict(focus=focus, type=ftype,
                         members={w: sorted(gg.id) for w, gg in g.groupby("wave")}))
    dates = c.waves["date"].tolist()
    events = []
    for e in c.events.to_dict("records"):
        valence = "negative" if e["type"] in NEGATIVE_EVENTS else "positive" if e["type"] in POSITIVE_EVENTS else "neutral"
        events.append(dict(date=str(e["date"].date()), type=e["type"], valence=valence,
                           actors=[a for a in e["actors"].split(";") if a],
                           third=[a for a in e["third_parties"].split(";") if a],
                           intensity=int(e["intensity"]), note=e.get("note", "")))
    R = {layer: np.rint(c.R[layer]).astype(int).tolist() for layer in LAYERS}

    # ------------------------------------------------------------------ forecast
    F = dict(fc)
    F["consensus"] = [[name(k) for k in g] for g in fc["consensus"]]
    F["couples_now"] = [[name(a), name(b)] for a, b in fc["couples_now"]]
    F["couple_survival"] = {"-".join(name(int(k)) for k in key.split("-")): v
                            for key, v in fc["couple_survival"].items()}
    F["triads"] = [dict(t, middle=name(t["middle"]), a=name(t["a"]), b=name(t["b"])) for t in fc["triads"]]
    F["romance_rates"] = {k: float(v) for k, v in fc["romance_rates"].items()}
    for key in ("p_positive", "p_negative", "p_dating", "comembership", "p_fight"):
        F[key] = np.round(np.array(fc[key]), 3).tolist()

    model = {k: v for k, v in fitted.items() if k != "beta_raw"}
    romance = romance_summary(c)

    findings = make_findings(c, W, model, romance, one_step, free_run)
    predictions = make_predictions(c, W, F)
    profiles = make_profiles(c, W, F)

    return dict(
        meta=dict(dataset=dataset_name, generated=dt.date.today().isoformat(), n=n,
                  synthetic="example" in dataset_name),
        ids=ids, waves=[dict(id=r.wave, label=r.label, date=str(r.date.date()), semester=r.semester)
                        for r in c.waves.itertuples()],
        persons=persons, person_wave=pw, foci=foci, events=events, R=R,
        counts=dict(persons=len(c.persons), person_wave=len(c.person_wave), foci=len(c.foci),
                    dyads=len(c.dyads), events=len(c.events)),
        issues=c.issues, W=W, layout=layout(c), model=model, romance=romance,
        validation=dict(one_step=one_step, free_run=free_run), forecast=F,
        findings=findings, predictions=predictions, profiles=profiles,
    )


# ---------------------------------------------------------------------- narrative
def make_findings(c, W, model, romance, one_step, free_run):
    first, last = W[0]["metrics"], W[-1]["metrics"]
    base = W[0]["baselines"]
    out = []
    out.append(dict(
        title="Contexts made the first friendships",
        body=(f"At {c.wave_ids[0]}, {_pct(first['focus_share'])} of friendship ties joined people who shared "
              f"a dorm, class section, chat or study group. Shared context is the clearest positive driver in the "
              f"tie-change model (+{model['coefs'][3]['beta']:.2f} rating points per SD)."),
        theory="Focus theory (Feld); propinquity (Festinger)"))
    out.append(dict(
        title="Gender lines blurred",
        body=(f"The E-I index for gender went from {first['ei_gender']:+.2f} to {last['ei_gender']:+.2f} "
              f"(random mixing would give {base['ei_gender']:+.2f}). Friendships started almost entirely "
              f"within gender and became nearly as mixed as chance by the end of semester 2."),
        theory="Homophily (McPherson et al.), weakening as contact grows"))
    stab = [w["metrics"]["stability"] for w in W[1:]]
    worst = max(range(1, len(W)), key=lambda t: W[t]["metrics"]["lost_ties"])
    out.append(dict(
        title="Friendships reshuffle every half-semester",
        body=(f"Only {_pct(min(stab))}–{_pct(max(stab))} of friendship ties carry over between waves (Jaccard). "
              f"The biggest loss came at {c.wave_ids[worst]}: "
              f"{W[worst]['metrics']['lost_ties']} ties lost, {W[worst]['metrics']['new_ties']} gained."),
        theory="Network churn; Dunbar's limit on close ties"))
    tension_peak = max(range(len(W)), key=lambda t: W[t]["metrics"]["tension"])
    mids = sorted({u["middle"] for w in W for u in w["unbalanced"]})
    out.append(dict(
        title="Drama follows romance, then spreads through side-taking",
        body=(f"Hostile ties went from 0 to {last['negative_ties']} after the breakup. Tension "
              f"(share of signed triangles that are unbalanced) peaked at {_pct(W[tension_peak]['metrics']['tension'])} "
              f"in {c.wave_ids[tension_peak]}. People caught between two sides at some point: {', '.join(mids)}."),
        theory="Structural balance (Heider); side-taking"))
    clear = sorted([k for k in model["coefs"] if k["clear"]], key=lambda k: -abs(k["beta"]))[:4]
    out.append(dict(
        title="What moves a friendship rating",
        body="Strongest clear effects: " + "; ".join(
            f"{k['label'].lower()} ({k['beta']:+.2f})" for k in clear) +
            f". The model explains {_pct(model['r2'])} of rating changes.",
        theory="Tie-change regression; stand-in for RSiena on the full data"))
    gaps = []
    F_last = c.R["friendship"][-1]
    for i in range(c.n):
        for j in range(i + 1, c.n):
            d = F_last[i, j] - F_last[j, i]
            if abs(d) >= 2:
                a, b = (i, j) if d > 0 else (j, i)
                gaps.append(f"{c.ids[a]} rates {c.ids[b]} {int(F_last[a, b])}, gets {int(F_last[b, a])} back")
    if gaps:
        out.append(dict(
            title="Unreciprocated ties are where people get hurt",
            body=f"Largest perception gaps at {c.wave_ids[-1]}: " + "; ".join(gaps) + ".",
            theory="Perception asymmetry; attachment"))
    s = one_step
    out.append(dict(
        title="The model beats 'nothing changes', but drama is hard to foresee",
        body=(f"Predicting each wave from the one before, the tie-level error (Brier) is "
              f"{np.mean([r['brier'] for r in s]):.2f} against {np.mean([r['brier_persistence'] for r in s]):.2f} "
              f"for assuming no change. Started from {c.wave_ids[0]} and run freely, it reproduces density, "
              f"clustering and gender mixing, but "
              + ("not the number of hostile ties: the breakup cascade is path-dependent."
                 if not free_run["negative_ties"]["inside"] else "also the number of hostile ties.")),
        theory="Back-testing"))
    return out


def make_predictions(c, W, F):
    ids, n = c.ids, c.n
    out = []
    pdat = np.array(F["p_dating"])
    current = {tuple(p) for p in F["couples_now"]}
    cands = sorted(((pdat[i, j], ids[i], ids[j]) for i in range(n) for j in range(i + 1, n)
                    if (ids[i], ids[j]) not in current), reverse=True)
    for p, a, b in cands[:2]:
        if p >= 0.05:
            out.append(dict(kind="romance", p=float(p), title=f"{a} and {b} start dating",
                            body="Mutual interest is already reported; both are single."
                            if p > 0.3 else "One-way interest at the last wave; needs to be returned first."))
    for key, v in F["couple_survival"].items():
        out.append(dict(kind="romance", p=float(1 - v), title=f"{key.replace('-', ' and ')} break up",
                        body=f"Breakup hazard estimated at {F['breakup_hazard_month']:.2f} per month "
                             f"(one breakup in {c.wave_ids[0]}–{c.wave_ids[-1]}, shrunk toward a prior)."))
    for t in F["triads"]:
        dropped, kept, p = ((t["b"], t["a"], t["drop_b"]) if t["drop_b"] >= t["drop_a"]
                            else (t["a"], t["b"], t["drop_a"]))
        out.append(dict(kind="balance", p=float(p),
                        title=f"{t['middle']} sides with {kept} and drifts from {dropped}",
                        body=(f"{t['middle']} is friends with both {t['a']} and {t['b']}, who are hostile. "
                              f"Still caught in the middle at the end of S3: {_pct(t['unresolved'])}; "
                              f"{t['a']} and {t['b']} reconcile: {_pct(t['reconciled'])}.")))
    piso = np.array(F["p_isolated"])
    for k in np.argsort(-piso)[:2]:
        out.append(dict(kind="isolation", p=float(piso[k]), title=f"{ids[k]} ends S3 with at most one friend",
                        body=f"Expected friendships at the end of S3: {F['expected_degree'][k]:.1f}."))
    pneg = np.array(F["p_negative"])
    S_last = np.array(W[-1]["S"])
    new_conf = sorted(((pneg[i, j], ids[i], ids[j]) for i in range(n) for j in range(i + 1, n)
                       if S_last[i, j] >= 0), reverse=True)
    for p, a, b in new_conf[:1]:
        if p >= 0.1:
            couple = (a, b) in current
            out.append(dict(kind="conflict", p=float(p), title=f"New hostility between {a} and {b}",
                            body="They are a couple now; this is the risk that a breakup turns hostile, "
                                 "as the last one did." if couple else
                                 "Not hostile at the last wave; the model gets there through rivalry or side-taking."))
    persist = sorted(((pneg[i, j], ids[i], ids[j]) for i in range(n) for j in range(i + 1, n)
                      if S_last[i, j] < 0), reverse=True)
    if persist:
        out.append(dict(kind="conflict", p=float(np.mean([p for p, _, _ in persist])),
                        title="Existing hostilities last through S3",
                        body="Average over the " + str(len(persist)) + " hostile pairs: " +
                             ", ".join(f"{a}–{b} {_pct(p)}" for p, a, b in persist) + "."))
    return out


def make_profiles(c, W, F):
    ids = c.ids
    out = {}
    persons = c.persons.set_index("id")
    for k, pid in enumerate(ids):
        p0, p1 = W[0]["people"][pid], W[-1]["people"][pid]
        r0, r1 = W[0]["roles"][pid], W[-1]["roles"][pid]
        bits = []
        fr = lambda d: f"{d} friend" + ("" if d == 1 else "s")
        bits.append(f"Started as {', '.join(r0)} ({fr(p0['degree'])} at {c.wave_ids[0]}); "
                    f"now {', '.join(r1)} ({fr(p1['degree'])} at {c.wave_ids[-1]}).")
        gap = p1["gives"] - p1["receives"]
        if gap >= 0.4:
            bits.append(f"Rates others {gap:.1f} points higher than they rate {pid}: invests more than is returned.")
        elif gap <= -0.4:
            bits.append(f"Others rate {pid} {-gap:.1f} points higher than {pid} rates them: sought after.")
        S = np.array(W[-1]["S"])
        enemies = [ids[j] for j in range(c.n) if S[k, j] < 0]
        if enemies:
            bits.append(f"Hostile with {', '.join(enemies)}.")
        rom = []
        for a, b in W[-1]["couples"]:
            if pid in (a, b):
                rom.append(f"dating {b if a == pid else a}")
        for a, b in W[-1]["exes"]:
            if pid in (a, b):
                rom.append(f"ex of {b if a == pid else a}")
        for a, b, m in W[-1]["crushes"]:
            if a == pid:
                rom.append(f"interested in {b}" + (" (mutual)" if m else ""))
        if rom:
            bits.append("Romance: " + "; ".join(rom) + ".")
        bits.append(f"Forecast for end of S3: {F['expected_degree'][k]:.1f} friends expected, "
                    f"{_pct(F['p_isolated'][k])} risk of isolation, {_pct(F['p_conflict'][k])} chance of at least one hostile tie.")
        out[pid] = " ".join(bits)
    return out


# ---------------------------------------------------------------------- output
def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(type(o))


def write(results, out_dir: Path, standalone=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, default=_default, separators=(",", ":"))
    (out_dir / "results.json").write_text(json.dumps(json.loads(payload), indent=1))
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", payload.replace("</", "<\\/"))
    if standalone:
        html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                '</head>\n<body>\n' + html + '\n</body>\n</html>\n')
    (out_dir / ("report.html" if standalone else "report.fragment.html")).write_text(html)

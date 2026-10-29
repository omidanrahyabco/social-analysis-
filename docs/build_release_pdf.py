"""Build the release PDF: figures from the pilot's results.json, text from the Markdown chapters.

    python run_analysis.py data/example_10p_2sem
    python docs/build_release_pdf.py

Writes docs/figures/*.png and docs/release/cohort-model-v<version>.pdf.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIG = DOCS / "figures"
RESULTS = ROOT / "outputs" / "example_10p_2sem" / "results.json"
VERSION = "0.1.0"
CHAPTERS = ["RELEASE_NOTES.md", "MODEL.md", "DATA_GUIDE.md", "PILOT_TEST.md"]

INK, INK2, MUTED, GRID, AXIS = "#101316", "#4a5159", "#80868d", "#e1e4e7", "#c2c7cc"
POS, NEG, ROM, ACC, C2, MID = "#2a78d6", "#e34948", "#e87ba4", "#1f5fa6", "#eb6834", "#f0efec"
GC = {1: "#eb6834", 2: "#1baf7a", 3: "#6250d6", 4: "#eda100", 5: "#008300", 0: "#9aa0a6"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def gnum(label):
    k = int(re.sub(r"\D", "", label) or 0)
    return k if 1 <= k <= 5 else 0


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)


# =========================================================================== figures
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_axis_off()
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 3.6)
    boxes = [
        ("Six CSV files", "waves, persons, person_wave,\nfoci, dyads, events"),
        ("Load and validate", "cohort/io.py\nIDs, ranges, duplicates;\nfills skipped answers"),
        ("Describe each wave", "cohort/network.py\nmeasures, balance, groups\ntracked across waves, roles"),
        ("Explain change", "cohort/models.py\ntie-change regression,\nbootstrap intervals"),
        ("Simulate", "cohort/simulate.py\none-step and free-run\nvalidation; forecast"),
        ("Report", "cohort/report.py\nfindings, predictions,\nprofiles; report.html"),
    ]
    w, h = 2.55, 1.25
    pos = [(0.2, 2.15), (3.2, 2.15), (6.2, 2.15), (6.2, 0.25), (3.2, 0.25), (0.2, 0.25)]
    for (title, sub), (x, y) in zip(boxes, pos):
        first = title == "Six CSV files"
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#eef3f9" if first else "white", ec=ACC if not first else AXIS, lw=1.3))
        ax.text(x + 0.14, y + h - 0.22, title, fontsize=10, fontweight="bold", color=INK, va="top")
        ax.text(x + 0.14, y + h - 0.52, sub, fontsize=7.6, color=INK2, va="top", linespacing=1.35)
    arrows = [((2.75, 2.78), (3.2, 2.78)), ((5.75, 2.78), (6.2, 2.78)), ((7.47, 2.15), (7.47, 1.5)),
              ((6.2, 0.88), (5.75, 0.88)), ((3.2, 0.88), (2.75, 0.88))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.4, mutation_scale=12))
    save(fig, "fig_pipeline.png")


def fig_contexts(r):
    ids, waves = r["ids"], [w["id"] for w in r["waves"]]
    foci = [f for f in r["foci"] if f["focus"] != "class-group"]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(foci) + 0.7))
    ax.set_axis_off()
    lw, cw = 3.2, 0.62
    for c, pid in enumerate(ids):
        ax.text(lw + c * cw + cw / 2 - 0.05, len(foci) + 0.15, pid, ha="center", fontsize=8, color=INK, family="DejaVu Sans Mono")
    for k, f in enumerate(foci):
        y = len(foci) - 1 - k
        ax.text(0, y + 0.42, f["focus"], fontsize=8.2, color=INK, va="center")
        ax.text(1.75, y + 0.42, f["type"], fontsize=7.2, color=MUTED, va="center")
        ax.plot([0, lw + len(ids) * cw], [y, y], color=GRID, lw=0.7)
        for c, pid in enumerate(ids):
            for t, w in enumerate(waves):
                on = pid in f["members"].get(w, [])
                ax.add_patch(Rectangle((lw + c * cw + t * 0.13, y + 0.22), 0.1, 0.42, color=ACC if on else GRID, lw=0))
    ax.set_xlim(0, lw + len(ids) * cw)
    ax.set_ylim(-0.1, len(foci) + 0.5)
    save(fig, "fig_contexts.png")


def fig_timeline(r):
    ev = r["events"]
    dates = [np.datetime64(e["date"]) for e in ev]
    fig, ax = plt.subplots(figsize=(9, 0.27 * len(ev) + 0.9))
    for w in r["waves"]:
        d = np.datetime64(w["date"])
        ax.axvline(d, color=AXIS, lw=1)
        ax.text(d, len(ev) + 0.1, w["id"], ha="center", fontsize=8, color=INK, fontweight="bold")
    for k, (e, d) in enumerate(zip(ev, dates)):
        y = len(ev) - 1 - k
        col = POS if e["valence"] == "positive" else NEG if e["valence"] == "negative" else MUTED
        ax.scatter([d], [y], s=22 + e["intensity"] * 16, color=col, zorder=3, edgecolor="white", linewidth=1.2)
        ax.text(d + np.timedelta64(7, "D"), y, f"{k + 1}. {e['type'].replace('_', ' ')} · {', '.join(e['actors'])}"
                + (f" (+{', '.join(e['third'])})" if e["third"] else ""), va="center", fontsize=7.3, color=INK2)
    ax.set_yticks([])
    ax.set_ylim(-0.8, len(ev) + 0.6)
    ax.set_xlim(np.datetime64("2024-09-20"), np.datetime64("2025-09-15"))
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    save(fig, "fig_timeline.png")


def _hull(ax, pts, color):
    ring = []
    for x, y in pts:
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            ring.append((x + 0.075 * np.cos(a), y + 0.075 * np.sin(a)))
    ring = np.array(ring)
    hull = ring[ConvexHull(ring).vertices]
    ax.add_patch(Polygon(hull, closed=True, fc=color, ec=color, alpha=0.14, lw=0))
    ax.add_patch(Polygon(hull, closed=True, fill=False, ec=color, alpha=0.6, lw=1))


def _draw_net(ax, r, P, edges, groups, title):
    persons = {p["id"]: p for p in r["persons"]}
    ax.set_title(title, pad=4)
    ax.set_axis_off()
    for g in groups:
        if len(g["members"]) >= 2:
            _hull(ax, [P[m] for m in g["members"]], GC[gnum(g["label"])])
    for e in edges:
        (x1, y1), (x2, y2) = P[e["a"]], P[e["b"]]
        if e.get("arrow"):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>", color=ROM, lw=1, shrinkA=9, shrinkB=9,
                                        connectionstyle="arc3,rad=0.25", mutation_scale=8))
        else:
            ax.plot([x1, x2], [y1, y2], color=e["c"], lw=e["w"], alpha=e.get("alpha", 0.85),
                    ls=e.get("ls", "-"), solid_capstyle="round", zorder=2)
    for pid, (x, y) in P.items():
        marker = "o" if persons[pid]["gender"] == "F" else "s"
        ax.scatter([x], [y], s=210, marker=marker, color="white", edgecolor=INK, linewidth=1.3, zorder=4)
        ax.text(x, y, pid, ha="center", va="center", fontsize=5.6, fontweight="bold", color=INK, zorder=5)
    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(-0.14, 1.14)
    ax.set_aspect("equal")


def fig_networks(r):
    ids = r["ids"]
    P = {p: (xy[0], 1 - xy[1]) for p, xy in r["layout"].items()}
    R, W, F = r["R"], r["W"], r["forecast"]
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.3))
    for t, ax in enumerate(axes.flat[:4]):
        w, edges = W[t], []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                s = w["S"][i][j]
                fm = (R["friendship"][t][i][j] + R["friendship"][t][j][i]) / 2
                if s > 0:
                    edges.append(dict(a=ids[i], b=ids[j], c=POS, w=0.8 + (fm - 3) * 1.1))
                elif s < 0:
                    edges.append(dict(a=ids[i], b=ids[j], c=NEG, w=1.6))
        edges += [dict(a=a, b=b, c=ROM, w=3.2) for a, b in w["couples"]]
        edges += [dict(a=a, b=b, c=ROM, w=1.2, ls=(0, (3, 2))) for a, b in w["exes"]]
        edges += [dict(a=a, b=b, arrow=True) for a, b, _ in w["crushes"]]
        wv = r["waves"][t]
        _draw_net(ax, r, P, edges, w["groups"], f"{wv['id']} · {wv['label']}")
    # forecast
    groups = []
    used = set()
    for g in F["consensus"]:
        best, bo = "G0", 1
        for h in W[-1]["groups"]:
            o = len(set(h["members"]) & set(g))
            if o > bo:
                best, bo = h["label"], o
        if best in used:
            best = "G0"
        used.add(best)
        groups.append(dict(label=best, members=g))
    edges = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pp, pn, pd = F["p_positive"][i][j], F["p_negative"][i][j], F["p_dating"][i][j]
            if pp >= 0.5:
                edges.append(dict(a=ids[i], b=ids[j], c=POS, w=0.6 + (pp - 0.5) * 5, alpha=0.3 + 0.6 * pp))
            if pn >= 0.5:
                edges.append(dict(a=ids[i], b=ids[j], c=NEG, w=1.6, alpha=0.3 + 0.6 * pn))
            if pd >= 0.5:
                edges.append(dict(a=ids[i], b=ids[j], c=ROM, w=3.2, alpha=0.4 + 0.5 * pd))
    _draw_net(axes[1, 1], r, P, edges, groups, "End of S3 · forecast (ties in ≥50% of runs)")
    ax = axes[1, 2]
    ax.set_axis_off()
    items = [(POS, 2, "-", "friendship (wider = stronger)"), (NEG, 1.6, "-", "hostility"),
             (ROM, 3.2, "-", "dating"), (ROM, 1.2, (0, (3, 2)), "exes"), (ROM, 1, "-", "interest (arrow)")]
    for k, (c, lw_, ls, lab) in enumerate(items):
        y = 0.94 - k * 0.1
        ax.plot([0.05, 0.22], [y, y], color=c, lw=lw_, ls=ls)
        ax.text(0.27, y, lab, va="center", fontsize=8, color=INK2)
    ax.scatter([0.135], [0.36], s=120, marker="s", color="white", edgecolor=INK)
    ax.text(0.27, 0.36, "man", va="center", fontsize=8, color=INK2)
    ax.scatter([0.135], [0.25], s=120, marker="o", color="white", edgecolor=INK)
    ax.text(0.27, 0.25, "woman", va="center", fontsize=8, color=INK2)
    ax.text(0.05, 0.03, "Shaded areas: groups found at that wave;\ncolour follows the group across waves.",
            fontsize=7.3, color=MUTED, va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.subplots_adjust(wspace=0.04, hspace=0.12)
    save(fig, "fig_networks.png")


def fig_metrics(r):
    W, base = r["W"], r["W"][-1]["baselines"]
    L = [w["id"] for w in r["waves"]]
    specs = [("density", "Density", 1, (0, .6)), ("transitivity", "Transitivity", 1, (0, .6)),
             ("positive_ties", "Friendship ties", 0, (0, 20)), ("negative_ties", "Hostile ties", 0, (0, 6)),
             ("tension", "Tension", 1, (0, .3)), ("ei_gender", "Gender E-I", 2, (-1, .5)),
             ("ei_housing", "Housing E-I", 2, (-1, .5)), ("stability", "Ties kept from last wave", 1, (0, 1))]
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.1))
    for ax, (k, title, kind, dom) in zip(axes.flat, specs):
        vals = [w["metrics"][k] for w in W]
        xs = [i for i, v in enumerate(vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.plot(xs, ys, color=ACC, lw=2, marker="o", ms=5, mec="white", mew=1.2)
        if k in base:
            ax.axhline(base[k], color=INK2, lw=0.9, ls=(0, (2, 2)))
            ax.text(3.15, base[k] + 0.04, "random", fontsize=6.5, color=MUTED, ha="right")
        fmt = (lambda v: f"{v:.0%}") if kind == 1 else (lambda v: f"{v:.0f}") if kind == 0 else (lambda v: f"{v:+.2f}")
        ax.text(xs[-1] + 0.12, ys[-1], fmt(ys[-1]), fontsize=7.5, color=INK, va="center")
        ax.set_title(title)
        ax.set_xticks(range(len(L)), L)
        ax.set_ylim(*dom)
        ax.set_xlim(-0.3, len(L) - 0.4)
        if kind == 1:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.locator_params(axis="y", nbins=4)
    fig.tight_layout(h_pad=1.6)
    save(fig, "fig_metrics.png")


def fig_groups(r):
    W, T = r["W"], len(r["W"])
    ids = r["ids"]
    cols = [dict(label=r["waves"][t]["id"], groups=W[t]["groups"]) for t in range(T)]
    fc, used = [], set()
    for g in r["forecast"]["consensus"]:
        best, bo = "G0", 1
        for h in W[-1]["groups"]:
            o = len(set(h["members"]) & set(g))
            if o > bo:
                best, bo = h["label"], o
        if best in used:
            best = "G0"
        used.add(best)
        fc.append(dict(label=best, members=g))
    cols.append(dict(label="S3 end (forecast)", groups=fc))
    ypos = [dict() for _ in cols]
    bands = []
    for k, c in enumerate(cols):
        def avg(g):
            return np.mean([ypos[k - 1][m] for m in g["members"]]) if k else -len(g["members"])
        order = sorted(c["groups"], key=avg)
        y = 0
        for g in order:
            mem = sorted(g["members"], key=lambda m: ypos[k - 1][m]) if k else g["members"]
            y0 = y
            for j, m in enumerate(mem):
                ypos[k][m] = y + 0.5 + j
            y += len(mem) + 0.7
            bands.append((k, g["label"], y0, len(mem)))
    fig, ax = plt.subplots(figsize=(9, 3.9))
    ax.set_axis_off()
    for k, lab, y0, h in bands:
        col = GC[gnum(lab)]
        ax.add_patch(Rectangle((k * 2 - 0.42, y0 + 0.05), 0.84, h - 0.1, color=col, alpha=0.16, lw=0))
        ax.add_patch(Rectangle((k * 2 - 0.42, y0 + 0.05), 0.05, h - 0.1, color=col, lw=0))
        ax.text(k * 2 + 0.38, y0 + 0.35, "new" if lab == "G0" else lab, fontsize=6.5, color=INK2, ha="right", va="center")
    for m in ids:
        xs, ys = [], []
        for k in range(len(cols)):
            xs += [k * 2 - 0.3, k * 2 + 0.25]
            ys += [ypos[k][m], ypos[k][m]]
        # smooth S-curves between columns
        px, py = [xs[0]], [ys[0]]
        for a in range(1, len(xs)):
            if a % 2 == 1:
                px.append(xs[a]); py.append(ys[a])
            else:
                tt = np.linspace(0, 1, 20)
                s = tt * tt * (3 - 2 * tt)
                px += list(xs[a - 1] + (xs[a] - xs[a - 1]) * tt)
                py += list(ys[a - 1] + (ys[a] - ys[a - 1]) * s)
        ax.plot(px, py, color=INK2, lw=1.4, alpha=0.85, solid_capstyle="round")
        ax.text(-0.5, ypos[0][m], m, ha="right", va="center", fontsize=7.5, family="DejaVu Sans Mono", color=INK)
        ax.text(len(cols) * 2 - 1.5, ypos[-1][m], m, ha="left", va="center", fontsize=7.5, family="DejaVu Sans Mono", color=INK)
    for k, c in enumerate(cols):
        ax.text(k * 2, -0.6, c["label"], ha="center", fontsize=8.5, fontweight="bold",
                color=ACC if k == len(cols) - 1 else INK)
    ax.set_xlim(-1.1, len(cols) * 2 - 1.1)
    ax.set_ylim(max(y for y in [max(p.values()) for p in ypos]) + 0.8, -1.2)
    save(fig, "fig_groups.png")


def fig_coefs(r):
    rows = sorted(r["model"]["coefs"], key=lambda c: c["beta"])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for k, c in enumerate(rows):
        col = (POS if c["beta"] > 0 else NEG) if c["clear"] else MUTED
        ax.plot([c["lo"], c["hi"]], [k, k], color=col, lw=2, solid_capstyle="round")
        ax.scatter([c["beta"]], [k], s=46, color=col, edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(c["hi"] + 0.03, k, f"{c['beta']:+.2f}", va="center", fontsize=7.3, color=INK2)
    ax.axvline(0, color=AXIS, lw=1.2)
    ax.set_yticks(range(len(rows)), [c["label"] for c in rows])
    ax.tick_params(axis="y", length=0, labelsize=8.2, labelcolor=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("effect on the change in a friendship rating (points per SD), 90% interval")
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.95, 0.75)
    save(fig, "fig_coefs.png")


def _band(ax, labels, bands, observed, title, fmt, dom, dashed=False, connect=False):
    xs = [i for i, b in enumerate(bands) if b]
    lo = [bands[i]["lo"] for i in xs]
    hi = [bands[i]["hi"] for i in xs]
    md = [bands[i]["median"] for i in xs]
    ax.fill_between(xs, lo, hi, color=ACC, alpha=0.15, lw=0)
    ax.plot(xs, md, color=ACC, lw=1.8, ls=(0, (4, 2)) if dashed else "-")
    ox = [i for i, v in enumerate(observed) if v is not None]
    oy = [v for v in observed if v is not None]
    if connect:
        ax.plot(ox, oy, color=ACC, lw=2)
    ax.scatter(ox, oy, s=26, color=INK if not connect else ACC, edgecolor="white", linewidth=1, zorder=3)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_title(title)
    ax.set_ylim(*dom)
    if fmt == "pct":
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.locator_params(axis="y", nbins=4)


def fig_validation(r):
    S = r["validation"]["one_step"]
    FR = r["validation"]["free_run"]
    L = [w["id"] for w in r["waves"]]
    fig = plt.figure(figsize=(9, 3.1))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    x = np.arange(len(S))
    ax.bar(x - 0.17, [s["brier"] for s in S], 0.3, color=ACC, label="model")
    ax.bar(x + 0.17, [s["brier_persistence"] for s in S], 0.3, color=MUTED, label="no change")
    for i, s in enumerate(S):
        ax.text(i - 0.17, s["brier"] + 0.005, f"{s['brier']:.2f}", ha="center", fontsize=7, color=INK)
        ax.text(i + 0.17, s["brier_persistence"] + 0.005, f"{s['brier_persistence']:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, [f"{s['from_wave']}→{s['to_wave']}" for s in S])
    ax.set_ylim(0, 0.26)
    ax.set_title("One-step error (Brier)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    for k, (key, title, fmt, dom) in enumerate([("positive_ties", "Friendship ties", "n", (0, 20)),
                                                ("negative_ties", "Hostile ties", "n", (0, 6)),
                                                ("density", "Density", "pct", (0, .6))]):
        _band(fig.add_subplot(gs[k + 1]), L, FR[key]["bands"], FR[key]["observed"], title, fmt, dom)
    save(fig, "fig_validation.png")


def fig_forecast(r):
    F, W, ids = r["forecast"], r["W"], r["ids"]
    n = len(ids)
    fig = plt.figure(figsize=(9, 4.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1, 1], wspace=0.42, hspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    mid = np.array(matplotlib.colors.to_rgb(MID))
    pos, neg = np.array(matplotlib.colors.to_rgb(POS)), np.array(matplotlib.colors.to_rgb(NEG))
    img = np.ones((n, n, 3))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p, q = F["p_positive"][i][j], F["p_negative"][i][j]
            img[i, j] = mid + (neg - mid) * q if q > p else mid + (pos - mid) * p
            v = max(p, q)
            if v >= 0.1:
                ax.text(j, i, f"{round(v * 100)}", ha="center", va="center", fontsize=6.2,
                        color="white" if v > 0.55 else INK)
    ax.imshow(img)
    ax.set_xticks(range(n), ids, rotation=90, fontsize=7)
    ax.set_yticks(range(n), ids, fontsize=7)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("End of S3, % of futures")
    L = [w["id"] for w in r["waves"]] + ["W5", "W6"]
    for k, (key, title, fmt, dom) in enumerate([("positive_ties", "Friendship ties", "n", (0, 20)),
                                                ("negative_ties", "Hostile ties", "n", (0, 7)),
                                                ("tension", "Tension", "pct", (0, .5)),
                                                ("density", "Density", "pct", (0, .6))]):
        a = fig.add_subplot(gs[k // 2, 1 + k % 2])
        obs = [w["metrics"][key] for w in W] + [None, None]
        last = W[-1]["metrics"][key]
        bands = [None] * (len(W) - 1) + [dict(lo=last, hi=last, median=last)] + F["trajectories"][key]
        _band(a, L, bands, obs, title, fmt, dom, dashed=True, connect=True)
        a.tick_params(axis="x", labelsize=6.8)
    save(fig, "fig_forecast.png")


def make_figures(r):
    fig_pipeline()
    fig_contexts(r)
    fig_timeline(r)
    fig_networks(r)
    fig_metrics(r)
    fig_groups(r)
    fig_coefs(r)
    fig_validation(r)
    fig_forecast(r)


# =========================================================================== PDF
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def register_fonts():
    pdfmetrics.registerFont(TTFont("Sans", str(FONT_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("Sans-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("Mono", str(FONT_DIR / "DejaVuSansMono.ttf")))
    pdfmetrics.registerFont(TTFont("Mono-Bold", str(FONT_DIR / "DejaVuSansMono-Bold.ttf")))
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans", boldItalic="Sans-Bold")
    pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-Bold", italic="Mono", boldItalic="Mono-Bold")


C_INK, C_INK2, C_MUTED, C_ACC = (colors.HexColor(c) for c in (INK, INK2, MUTED, ACC))
C_GRID, C_WASH = colors.HexColor(GRID), colors.HexColor("#eef3f9")

ST = dict(
    body=ParagraphStyle("body", fontName="Sans", fontSize=9.4, leading=14, textColor=C_INK, spaceAfter=6),
    h1=ParagraphStyle("h1", fontName="Sans-Bold", fontSize=21, leading=25, textColor=C_INK, spaceAfter=12),
    h2=ParagraphStyle("h2", fontName="Sans-Bold", fontSize=13.5, leading=17, textColor=C_INK, spaceBefore=12, spaceAfter=6, keepWithNext=1),
    h3=ParagraphStyle("h3", fontName="Sans-Bold", fontSize=10.5, leading=14, textColor=C_ACC, spaceBefore=8, spaceAfter=4, keepWithNext=1),
    bullet=ParagraphStyle("bullet", fontName="Sans", fontSize=9.4, leading=13.6, textColor=C_INK, leftIndent=14,
                          bulletIndent=3, spaceAfter=3),
    cell=ParagraphStyle("cell", fontName="Sans", fontSize=7.9, leading=10.4, textColor=C_INK),
    head=ParagraphStyle("head", fontName="Sans-Bold", fontSize=7.6, leading=10, textColor=C_INK2),
    code=ParagraphStyle("code", fontName="Mono", fontSize=7.6, leading=10.2, textColor=C_INK),
    cap=ParagraphStyle("cap", fontName="Sans", fontSize=8, leading=11, textColor=C_INK2, spaceBefore=3, spaceAfter=10),
    toc1=ParagraphStyle("toc1", fontName="Sans-Bold", fontSize=10.5, leading=16, leftIndent=0),
    toc2=ParagraphStyle("toc2", fontName="Sans", fontSize=9, leading=13, leftIndent=16, textColor=C_INK2),
)


def inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Mono" size="8.3" color="#1f5fa6">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"\1", text)
    text = text.replace("\\|", "|")
    return text


class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2.1 * cm,
                         bottomMargin=2 * cm, title=f"Cohort Social Dynamics Model v{VERSION}",
                         author="social-analysis- project", subject="Model, data guide and pilot test", **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=self.cover),
                               PageTemplate("body", [frame], onPageEnd=self.chrome)])
        self.meta = {}

    def cover(self, c, doc):
        w, h = A4
        c.saveState()
        c.setFillColor(C_ACC)
        c.rect(0, h - 7.2 * cm, w, 7.2 * cm, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Mono", 9)
        c.drawString(2 * cm, h - 2.2 * cm, f"RELEASE v{VERSION} · {self.meta['date']}")
        c.setFont("Sans-Bold", 30)
        c.drawString(2 * cm, h - 3.9 * cm, "Cohort Social Dynamics Model")
        c.setFont("Sans", 13)
        c.drawString(2 * cm, h - 4.9 * cm, "A time-aware model of friendship, groups, romance and conflict")
        c.drawString(2 * cm, h - 5.6 * cm, "in a university cohort")
        c.setFillColor(C_INK)
        c.setFont("Sans-Bold", 11)
        y = h - 9.2 * cm
        for line in ["How the model works", "How to put your data in", "The pilot test: a 10-student, 2-semester invented cohort",
                     "What the model found, and how it compares with the story"]:
            c.setFillColor(C_ACC)
            c.rect(2 * cm, y - 0.05 * cm, 0.18 * cm, 0.42 * cm, fill=1, stroke=0)
            c.setFillColor(C_INK)
            c.drawString(2.6 * cm, y, line)
            y -= 0.9 * cm
        c.setFont("Sans", 9)
        c.setFillColor(C_INK2)
        y = 6.2 * cm
        for k, v in [("Context", "AUT Mechanical Engineering, cohort entering Mehr 1403"),
                     ("Repository", "github.com/omidanrahyabco/social-analysis-"),
                     ("Source", f"branch {self.meta['branch']}, commit {self.meta['commit']}")]:
            c.setFont("Sans-Bold", 9)
            c.drawString(2 * cm, y, k)
            c.setFont("Sans", 9)
            c.drawString(4.6 * cm, y, v)
            y -= 0.6 * cm
        c.setStrokeColor(colors.HexColor("#b4361f"))
        c.setFillColor(colors.HexColor("#b4361f"))
        c.roundRect(2 * cm, 2.2 * cm, w - 4 * cm, 1.5 * cm, 4, fill=0, stroke=1)
        c.setFont("Sans-Bold", 8.5)
        c.drawString(2.4 * cm, 3.15 * cm, "SYNTHETIC DATA")
        c.setFont("Sans", 8.5)
        c.drawString(2.4 * cm, 2.62 * cm, "Every person, rating and event in the pilot test is invented to test the pipeline.")
        c.restoreState()

    def chrome(self, c, doc):
        w, h = A4
        c.saveState()
        c.setStrokeColor(C_GRID)
        c.line(2 * cm, h - 1.45 * cm, w - 2 * cm, h - 1.45 * cm)
        c.setFont("Sans", 7.5)
        c.setFillColor(C_MUTED)
        c.drawString(2 * cm, h - 1.25 * cm, f"Cohort Social Dynamics Model · v{VERSION}")
        c.drawRightString(w - 2 * cm, h - 1.25 * cm, self.meta.get("chapter", ""))
        c.drawRightString(w - 2 * cm, 1.2 * cm, str(doc.page))
        c.restoreState()

    def beforeDocument(self):
        self.meta["chapter"] = ""

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            text = f.getPlainText()
            level = 0 if f.style.name == "h1" else 1
            if level == 0:
                self.meta["chapter"] = text
            key = f"h{id(f)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


def md_table(lines, width):
    rows = [[c.strip() for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))] for ln in lines]
    header, body = rows[0], [r for r in rows[2:]]
    ncol = len(header)
    lens = np.array([[len(re.sub(r"[`*]", "", c)) for c in (r + [""] * ncol)[:ncol]] for r in [header] + body], float)
    longest = np.array([max(len(w) for r in [header] + body for w in re.sub(r"[`*]", "", ((r + [""] * ncol)[:ncol])[c]).split() or [""])
                        for c in range(ncol)], float)
    est = np.clip(np.maximum(np.percentile(lens, 85, axis=0) + 4, longest * 1.25 + 2), 6, 70)
    widths = est / est.sum() * width
    data = [[Paragraph(inline(c), ST["head"]) for c in header]]
    data += [[Paragraph(inline(c), ST["cell"]) for c in (r + [""] * ncol)[:ncol]] for r in body]
    t = Table(data, colWidths=list(widths), repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.9, colors.HexColor(AXIS)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, C_GRID),
        ("BACKGROUND", (0, 0), (-1, 0), C_WASH),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def md_to_flowables(path: Path, width, first_chapter=False):
    lines = path.read_text().splitlines()
    out, para, k = [], [], 0

    def flush():
        if para:
            out.append(Paragraph(inline(" ".join(para)), ST["body"]))
            para.clear()

    while k < len(lines):
        ln = lines[k]
        s = ln.strip()
        if s.startswith("```"):
            flush()
            block = []
            k += 1
            while k < len(lines) and not lines[k].strip().startswith("```"):
                block.append(lines[k])
                k += 1
            pre = Preformatted("\n".join(block), ST["code"])
            box = Table([[pre]], colWidths=[width], hAlign="LEFT")
            box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f6f8")),
                                     ("BOX", (0, 0), (-1, -1), 0.5, C_GRID),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                                     ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            out += [box, Spacer(1, 8)]
        elif s.startswith("# "):
            flush()
            if not first_chapter or out:
                pass
            out.append(Paragraph(inline(s[2:]), ST["h1"]))
        elif s.startswith("## "):
            flush()
            out.append(Paragraph(inline(s[3:]), ST["h2"]))
        elif s.startswith("### "):
            flush()
            out.append(Paragraph(inline(s[4:]), ST["h3"]))
        elif s.startswith("!["):
            flush()
            m = re.match(r"!\[(.*)\]\((.*)\)", s)
            img_path = path.parent / m.group(2)
            from reportlab.lib.utils import ImageReader
            iw, ih = ImageReader(str(img_path)).getSize()
            w_ = width
            h_ = w_ * ih / iw
            if h_ > 19 * cm:
                h_ = 19 * cm
                w_ = h_ * iw / ih
            out.append(KeepTogether([Image(str(img_path), width=w_, height=h_), Paragraph(inline(m.group(1)), ST["cap"])]))
        elif s.startswith("|"):
            flush()
            block = []
            while k < len(lines) and lines[k].strip().startswith("|"):
                block.append(lines[k])
                k += 1
            out += [md_table(block, width), Spacer(1, 8)]
            continue
        elif re.match(r"^(- |\d+\. )", s):
            flush()
            m = re.match(r"^(- \[ \] |- |\d+\. )(.*)", s)
            mark = m.group(1)
            text = m.group(2)
            while k + 1 < len(lines) and lines[k + 1].startswith("   ") and lines[k + 1].strip():
                k += 1
                text += " " + lines[k].strip()
            bullet = "☐" if "[ ]" in mark else "•" if mark.startswith("-") else mark.strip()
            out.append(Paragraph(inline(text), ST["bullet"], bulletText=bullet))
        elif s.startswith("> "):
            flush()
            out.append(Paragraph(inline(s[2:]), ST["cap"]))
        elif not s:
            flush()
        else:
            para.append(s)
        k += 1
    flush()
    return out


def build_pdf(out_path: Path):
    register_fonts()
    doc = Doc(out_path)
    doc.meta["date"] = dt.date.today().isoformat()
    try:
        doc.meta["commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
        doc.meta["branch"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        doc.meta["commit"] = doc.meta["branch"] = "unknown"
    story = [NextPageTemplate("body"), PageBreak()]
    toc = TableOfContents()
    toc.levelStyles = [ST["toc1"], ST["toc2"]]
    story += [Paragraph("Contents", ParagraphStyle("ct", parent=ST["h1"])), toc, PageBreak()]
    for k, name in enumerate(CHAPTERS):
        if k:
            story.append(PageBreak())
        story += md_to_flowables(DOCS / name, doc.width)
    doc.multiBuild(story)


def main():
    if not RESULTS.exists():
        sys.exit(f"{RESULTS} not found: run `python run_analysis.py data/example_10p_2sem` first")
    r = json.loads(RESULTS.read_text())
    make_figures(r)
    out = DOCS / "release" / f"cohort-model-v{VERSION}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_pdf(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

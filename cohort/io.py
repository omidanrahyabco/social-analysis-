"""Load and validate a cohort dataset (the six CSV files described in README.md)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LAYERS = {"friendship": 5, "trust": 5, "hostility": 5, "romantic": 3, "contact": 5}
FILES = {
    "waves": ["wave", "label", "date", "semester"],
    "persons": ["id", "gender", "origin", "attractiveness_peer"],
    "person_wave": ["id", "wave", "housing", "gpa"],
    "foci": ["id", "focus", "type", "wave"],
    "dyads": ["rater", "target", "wave", "layer", "value"],
    "events": ["date", "type", "actors", "third_parties", "intensity"],
}

@dataclass
class Cohort:
    waves: pd.DataFrame
    persons: pd.DataFrame
    person_wave: pd.DataFrame
    foci: pd.DataFrame
    dyads: pd.DataFrame
    events: pd.DataFrame
    ids: list[str]
    wave_ids: list[str]
    R: dict[str, np.ndarray]          # layer -> (T, n, n) ratings, rater rows, target cols
    shared_foci: np.ndarray           # (T, n, n) count of shared non-universal foci
    issues: list[dict] = field(default_factory=list)

    @property
    def n(self):
        return len(self.ids)

    @property
    def T(self):
        return len(self.wave_ids)

    def attr(self, column, wave=None):
        """Person attribute as an array aligned with self.ids."""
        if column in self.persons.columns:
            return self.persons.set_index("id").loc[self.ids, column].to_numpy()
        pw = self.person_wave[self.person_wave.wave == wave].set_index("id")
        return pw.loc[self.ids, column].to_numpy()


def load(folder: str | Path) -> Cohort:
    folder = Path(folder)
    issues: list[dict] = []
    frames = {}
    for name, required in FILES.items():
        df = pd.read_csv(folder / f"{name}.csv", dtype=str, keep_default_na=False)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name}.csv is missing columns: {missing}")
        frames[name] = df

    waves = frames["waves"].copy()
    waves["date"] = pd.to_datetime(waves["date"])
    waves = waves.sort_values("date").reset_index(drop=True)
    wave_ids = waves["wave"].tolist()

    persons = frames["persons"].copy()
    for c in persons.columns:
        if c not in ("id", "gender", "origin"):
            persons[c] = pd.to_numeric(persons[c], errors="coerce")
    ids = sorted(persons["id"].tolist())
    idx = {p: k for k, p in enumerate(ids)}
    widx = {w: k for k, w in enumerate(wave_ids)}

    person_wave = frames["person_wave"].copy()
    person_wave["gpa"] = pd.to_numeric(person_wave["gpa"], errors="coerce")

    foci = frames["foci"]
    events = frames["events"].copy()
    events["date"] = pd.to_datetime(events["date"])
    events["intensity"] = pd.to_numeric(events["intensity"])
    events = events.sort_values("date").reset_index(drop=True)

    dyads = frames["dyads"].copy()
    dyads["value"] = pd.to_numeric(dyads["value"], errors="coerce")

    # ---- validation
    for name, col in (("dyads", "rater"), ("dyads", "target"), ("foci", "id"), ("person_wave", "id")):
        unknown = sorted(set(frames[name][col]) - set(ids))
        if unknown:
            issues.append(dict(level="error", file=name, msg=f"unknown ids in {col}: {unknown}"))
    bad_layer = sorted(set(dyads.layer) - set(LAYERS))
    if bad_layer:
        issues.append(dict(level="error", file="dyads", msg=f"unknown layers: {bad_layer}"))
    self_rows = dyads[dyads.rater == dyads.target]
    if len(self_rows):
        issues.append(dict(level="warning", file="dyads", msg=f"{len(self_rows)} self-ratings dropped"))
        dyads = dyads[dyads.rater != dyads.target]
    dup = dyads.duplicated(["rater", "target", "wave", "layer"], keep="last")
    if dup.any():
        issues.append(dict(level="warning", file="dyads", msg=f"{int(dup.sum())} duplicate ratings, kept the last"))
        dyads = dyads[~dup]
    for layer, top in LAYERS.items():
        out = dyads[(dyads.layer == layer) & ((dyads.value < 0) | (dyads.value > top))]
        if len(out):
            issues.append(dict(level="error", file="dyads", msg=f"{len(out)} {layer} values outside 0-{top}"))
    if any(i["level"] == "error" for i in issues):
        raise ValueError("; ".join(i["msg"] for i in issues if i["level"] == "error"))

    # ---- rating tensors, with imputation of skipped answers
    n, T = len(ids), len(wave_ids)
    R = {}
    for layer in LAYERS:
        A = np.full((T, n, n), np.nan)
        d = dyads[dyads.layer == layer]
        A[d.wave.map(widx).to_numpy(), d.rater.map(idx).to_numpy(), d.target.map(idx).to_numpy()] = d.value.to_numpy()
        R[layer] = A
    missing_cells = []
    for t in range(T):
        for i in range(n):
            for j in range(n):
                if i != j and np.isnan(R["friendship"][t, i, j]):
                    missing_cells.append((wave_ids[t], ids[i], ids[j]))
    if missing_cells:
        issues.append(dict(level="warning", file="dyads",
                           msg=f"{len(missing_cells)} skipped answers imputed from the other person's rating of the same tie",
                           detail=[f"{w}: {a} about {b}" for w, a, b in missing_cells]))
    for layer, A in R.items():
        for t in range(T):
            M = A[t]
            gap = np.isnan(M)
            np.fill_diagonal(gap, False)
            M[gap] = M.T[gap]                       # reciprocal rating
            still = np.isnan(M)
            np.fill_diagonal(still, False)
            if still.any():
                M[still] = np.nanmedian(M)          # nobody rated it: cohort median
            np.fill_diagonal(M, 0)

    # ---- shared foci
    shared = np.zeros((T, n, n))
    for (focus, wave), g in foci.groupby(["focus", "wave"]):
        if wave not in widx:
            continue
        members = [idx[p] for p in g.id if p in idx]
        if len(members) >= n:                      # everyone is in it: no information
            continue
        for a in members:
            for b in members:
                if a != b:
                    shared[widx[wave], a, b] += 1

    return Cohort(waves, persons, person_wave, foci, dyads, events, ids, wave_ids, R, shared, issues)

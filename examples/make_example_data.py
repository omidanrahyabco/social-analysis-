"""Generate the FAKE example dataset: 10 students, 2 semesters (4 survey waves).

This script plays the role of the humans filling in the survey. It encodes an
invented storyline (who befriends whom, a romance, a breakup, side-taking) as
"true" relationship levels, then turns them into what people would actually
report: each person rates the other 9, with personal rating bias, random
recall noise, perception gaps and a few skipped answers.

Nothing here is used by the analysis. The pipeline (run_analysis.py) only reads
the CSV files written to data/example_10p_2sem/, exactly as it will read the
real cohort data.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "example_10p_2sem"
SEED = 1403
WAVES = ["W1", "W2", "W3", "W4"]

# --------------------------------------------------------------------------- people
# id, gender, origin, entry_age, BFI-10 scores (1-5), peer-rated attractiveness (1-5)
PERSONS = [
    # id    g    origin     age  E    A    C    N    O    attr
    ("P01", "M", "Isfahan", 18, 4.6, 3.8, 2.6, 2.5, 3.4, 3.2),
    ("P02", "M", "Tabriz", 18, 2.1, 3.5, 4.5, 2.8, 3.9, 2.9),
    ("P03", "M", "Tehran", 18, 4.5, 2.6, 2.4, 3.0, 3.2, 4.4),
    ("P04", "M", "Tehran", 18, 3.4, 4.1, 3.5, 2.2, 3.0, 3.3),
    ("P05", "M", "Mashhad", 19, 3.9, 4.3, 3.6, 2.0, 3.5, 3.0),
    ("P06", "M", "Tehran", 18, 1.8, 3.6, 4.8, 3.4, 4.2, 3.1),
    ("P07", "F", "Shiraz", 18, 4.2, 4.4, 4.2, 2.6, 4.0, 3.6),
    ("P08", "F", "Tehran", 18, 4.0, 2.9, 3.1, 3.9, 3.3, 4.5),
    ("P09", "F", "Rasht", 18, 3.8, 3.1, 2.9, 3.3, 3.6, 4.0),
    ("P10", "F", "Tehran", 18, 2.4, 3.9, 4.6, 2.9, 3.8, 3.4),
]
IDS = [p[0] for p in PERSONS]
BIG5 = {p[0]: dict(E=p[4], A=p[5], C=p[6], N=p[7], O=p[8]) for p in PERSONS}

# housing per wave and semester GPA (out of 20)
HOUSING = {
    "P01": ["dorm"] * 4, "P02": ["dorm", "dorm", "rented", "rented"],
    "P03": ["family"] * 4, "P04": ["family"] * 4,
    "P05": ["dorm", "dorm", "rented", "rented"], "P06": ["family"] * 4,
    "P07": ["dorm"] * 4, "P08": ["family"] * 4, "P09": ["dorm"] * 4, "P10": ["family"] * 4,
}
GPA = {  # S1, S2
    "P01": (15.1, 13.8), "P02": (17.9, 18.2), "P03": (14.2, 13.1), "P04": (15.6, 15.9),
    "P05": (15.0, 15.4), "P06": (18.6, 18.9), "P07": (17.4, 17.8), "P08": (15.4, 14.6),
    "P09": (14.8, 14.5), "P10": (18.1, 18.4),
}

# --------------------------------------------------------------------------- contexts
# focus_id, type, members, waves active
FOCI = [
    ("class-group", "online", IDS, WAVES),
    ("dorm-A", "dorm", ["P01", "P02", "P05"], ["W1", "W2"]),
    ("dorm-A", "dorm", ["P01"], ["W3", "W4"]),
    ("dorm-B", "dorm", ["P07", "P09"], WAVES),
    ("flat-1", "rented", ["P02", "P05"], ["W3", "W4"]),
    ("statics-sec1", "class", ["P01", "P03", "P04", "P05", "P08"], ["W1", "W2"]),
    ("statics-sec2", "class", ["P02", "P06", "P07", "P09", "P10"], ["W1", "W2"]),
    ("dynamics-sec1", "class", ["P01", "P02", "P03", "P05", "P09"], ["W3", "W4"]),
    ("dynamics-sec2", "class", ["P04", "P06", "P07", "P08", "P10"], ["W3", "W4"]),
    ("gaming-chat", "online", ["P01", "P03", "P04", "P05"], ["W1", "W2", "W3"]),
    ("study-group", "study", ["P02", "P06", "P07", "P10"], ["W2", "W3", "W4"]),
    ("mech-association", "extracurricular", ["P04", "P05", "P07"], ["W2", "W3", "W4"]),
]

# --------------------------------------------------------------------------- the story
# Symmetric "true" friendship 0-5 for W1..W4; unlisted pairs default to 1 (classmates).
FRIEND = {
    ("P01", "P02"): [4, 3, 2, 1], ("P01", "P05"): [4, 4, 3, 2], ("P02", "P05"): [3, 4, 5, 5],
    ("P03", "P04"): [4, 5, 4, 2], ("P01", "P03"): [3, 4, 3, 1], ("P01", "P04"): [2, 3, 3, 2],
    ("P03", "P05"): [2, 3, 2, 1], ("P04", "P05"): [2, 3, 3, 3], ("P04", "P06"): [1, 1, 2, 3],
    ("P03", "P06"): [1, 1, 1, 0], ("P07", "P09"): [4, 5, 4, 3], ("P08", "P10"): [4, 5, 5, 5],
    ("P07", "P10"): [1, 3, 4, 3], ("P08", "P09"): [2, 3, 1, 0], ("P07", "P08"): [1, 2, 2, 1],
    ("P09", "P10"): [1, 2, 1, 0], ("P02", "P06"): [2, 4, 4, 5], ("P06", "P10"): [1, 3, 4, 4],
    ("P02", "P07"): [1, 3, 4, 3], ("P06", "P07"): [1, 3, 4, 3], ("P02", "P10"): [1, 2, 3, 4],
    ("P03", "P08"): [3, 4, 0, 0], ("P03", "P09"): [1, 1, 3, 4], ("P01", "P07"): [1, 2, 2, 2],
    ("P01", "P09"): [1, 1, 2, 2], ("P05", "P07"): [1, 2, 3, 3], ("P05", "P09"): [1, 1, 2, 2],
    ("P04", "P08"): [2, 3, 4, 4], ("P04", "P10"): [1, 2, 3, 3], ("P04", "P07"): [1, 1, 2, 2],
    ("P03", "P10"): [1, 2, 0, 0], ("P02", "P03"): [1, 1, 1, 1], ("P01", "P06"): [0, 1, 1, 1],
}
# Directed hostility 0-5 (rater -> target)
HOSTILE = {
    ("P08", "P09"): [0, 0, 4, 4], ("P09", "P08"): [0, 0, 3, 3],
    ("P08", "P03"): [0, 0, 4, 3], ("P03", "P08"): [0, 0, 2, 2],
    ("P03", "P04"): [0, 0, 1, 3], ("P04", "P03"): [0, 0, 1, 2],
    ("P10", "P09"): [0, 0, 2, 3], ("P09", "P10"): [0, 0, 1, 2],
    ("P10", "P03"): [0, 0, 2, 2], ("P03", "P10"): [0, 0, 0, 1],
    ("P01", "P07"): [0, 1, 1, 0], ("P10", "P07"): [0, 0, 0, 2],
    ("P01", "P02"): [0, 0, 1, 1], ("P01", "P05"): [0, 0, 0, 1],
}
# Directed romantic: 0 none, 1 crush/interest, 2 dating, 3 ex
ROMANTIC = {
    ("P03", "P08"): [1, 2, 3, 3], ("P08", "P03"): [1, 2, 3, 3],
    ("P03", "P09"): [0, 0, 1, 2], ("P09", "P03"): [0, 1, 1, 2],
    ("P01", "P07"): [1, 1, 0, 0],
    ("P06", "P10"): [0, 0, 1, 1], ("P10", "P06"): [0, 0, 0, 1],
    ("P05", "P09"): [0, 0, 0, 1],
}
# Perception gaps: rater sees the tie differently from the "truth" (added to friendship)
PERCEPTION = {
    ("P01", "P02"): [0, 1, 1, 2], ("P01", "P03"): [0, 0, 1, 2], ("P01", "P05"): [0, 0, 1, 1],
    ("P09", "P07"): [0, 0, 1, 1], ("P04", "P08"): [0, 1, 1, 1], ("P07", "P10"): [0, 0, 0, 1],
    ("P03", "P04"): [0, 0, 1, 1], ("P05", "P09"): [0, 0, 1, 1],
}
EVENTS = [
    # date, type, actors, third parties, intensity (1-5), note
    ("2024-10-05", "group_formed", "P01;P03;P04;P05", "", 2, "gaming chat formed in first weeks"),
    ("2024-11-02", "started_dating", "P03;P08", "", 3, ""),
    ("2024-12-10", "group_formed", "P02;P06;P07;P10", "", 2, "statics study group before midterms"),
    ("2025-02-03", "crush_rejected", "P01;P07", "", 2, "between semesters"),
    ("2025-02-10", "moved_housing", "P02;P05", "", 1, "left dorm, share a rented flat"),
    ("2025-03-05", "gossip", "P03;P09", "P08;P04", 2, "seen together often in dynamics class"),
    ("2025-04-06", "breakup", "P03;P08", "P09;P04;P10", 5, "right after Nowruz break"),
    ("2025-04-12", "public_fight", "P08;P09", "P07;P10", 4, ""),
    ("2025-05-10", "fight", "P03;P04", "P08", 3, "P04 defends P08"),
    ("2025-05-15", "group_dissolved", "P01;P03;P04;P05", "", 2, "gaming chat goes silent"),
    ("2025-05-25", "started_dating", "P03;P09", "P08", 3, ""),
    ("2025-06-10", "cold_shoulder", "P10;P07", "P09", 2, "P07 still rooms with P09"),
    ("2025-06-20", "support", "P06;P10", "", 2, "exam prep help"),
]


def pair(i, j):
    return (i, j) if (i, j) in FRIEND else (j, i)


def clip(v, lo, hi):
    return max(lo, min(hi, int(round(v))))


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    with open(OUT / "waves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wave", "label", "date", "semester"])
        w.writerows([
            ["W1", "S1 mid (Aban 1403)", "2024-11-15", "S1"],
            ["W2", "S1 end (Bahman 1403)", "2025-01-25", "S1"],
            ["W3", "S2 mid (Farvardin 1404)", "2025-04-20", "S2"],
            ["W4", "S2 end (Tir 1404)", "2025-06-28", "S2"],
        ])

    with open(OUT / "persons.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "gender", "origin", "entry_age", "extraversion", "agreeableness",
                    "conscientiousness", "neuroticism", "openness", "attractiveness_peer"])
        for p in PERSONS:
            w.writerow(p)

    with open(OUT / "person_wave.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "wave", "housing", "gpa", "status"])
        for pid in IDS:
            for t, wave in enumerate(WAVES):
                w.writerow([pid, wave, HOUSING[pid][t], GPA[pid][t // 2], "active"])

    with open(OUT / "foci.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "focus", "type", "wave"])
        for focus, ftype, members, waves in FOCI:
            for wave in waves:
                for pid in members:
                    w.writerow([pid, focus, ftype, wave])

    shared = {}
    for focus, ftype, members, waves in FOCI:
        if focus == "class-group":
            continue
        for wave in waves:
            for a in members:
                for b in members:
                    if a != b:
                        shared[(a, b, wave)] = shared.get((a, b, wave), 0) + 1

    rows = []
    for t, wave in enumerate(WAVES):
        for i in IDS:
            for j in IDS:
                if i == j:
                    continue
                base_f = FRIEND.get(pair(i, j), [1, 1, 1, 1])[t]
                h = HOSTILE.get((i, j), [0] * 4)[t]
                rom = ROMANTIC.get((i, j), [0] * 4)[t]
                bias = (BIG5[i]["E"] - 3) * 0.25          # extraverts rate generously
                f = base_f + PERCEPTION.get((i, j), [0] * 4)[t] + bias
                if rng.random() < 0.22:
                    f += rng.choice([-1, 1])               # recall noise
                f = clip(f, 0, 5)
                if h > 0 and rng.random() < 0.2:
                    h += rng.choice([-1, 1])
                elif h == 0 and rng.random() < 0.03:
                    h = 1                                  # minor irritations nobody planned
                h = clip(h, 0, 5)
                trust = f * 0.85 - h * 0.8 + (BIG5[i]["A"] - 3.5) * 0.6
                if (i, j) == ("P08", "P03") and t >= 2:
                    trust = 0
                if rng.random() < 0.2:
                    trust += rng.choice([-1, 1])
                trust = clip(trust, 0, 5)
                s = shared.get((i, j, wave), 0)
                contact = 0.6 * base_f + 1.0 * s + 0.5
                if rom == 2:
                    contact = 5
                elif rom == 3:
                    contact = min(contact, 1)
                contact = clip(contact, 0, 5)
                for layer, val in (("friendship", f), ("trust", trust), ("hostility", h),
                                   ("romantic", rom), ("contact", contact)):
                    rows.append([i, j, wave, layer, val])

    # a few skipped answers, as in any real survey
    skipped = {("P06", "P08", "W1"), ("P06", "P09", "W1"), ("P03", "P10", "W3")}
    rows = [r for r in rows if (r[0], r[1], r[2]) not in skipped]

    with open(OUT / "dyads.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rater", "target", "wave", "layer", "value"])
        w.writerows(rows)

    with open(OUT / "events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "type", "actors", "third_parties", "intensity", "note"])
        w.writerows(EVENTS)

    print(f"wrote {len(rows)} dyad ratings to {OUT}")


if __name__ == "__main__":
    main()

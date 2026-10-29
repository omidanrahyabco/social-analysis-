"""Turn survey matrices into dyads.csv.

Surveys are easiest to fill in as one square sheet per layer and wave: the first
column is the person answering, the header row lists the people being rated.
Save each sheet as <layer>_<wave>.csv in one folder (for example friendship_W1.csv,
hostility_W3.csv) and run:

    python -m cohort.convert path/to/matrices path/to/dataset/dyads.csv

Empty cells are treated as skipped answers and left out; the loader fills them later.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .io import LAYERS


def matrices_to_long(folder: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(folder).glob("*.csv")):
        layer, _, wave = f.stem.partition("_")
        if layer not in LAYERS or not wave:
            print(f"skipped {f.name}: name it <layer>_<wave>.csv with layer one of {list(LAYERS)}")
            continue
        m = pd.read_csv(f, dtype=str, keep_default_na=False)
        m = m.set_index(m.columns[0])
        for rater, row in m.iterrows():
            for target, value in row.items():
                if str(rater).strip() == str(target).strip() or str(value).strip() == "":
                    continue
                rows.append(dict(rater=str(rater).strip(), target=str(target).strip(), wave=wave,
                                 layer=layer, value=int(float(value))))
    return pd.DataFrame(rows, columns=["rater", "target", "wave", "layer", "value"])


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    df = matrices_to_long(Path(sys.argv[1]))
    df.to_csv(sys.argv[2], index=False)
    print(f"wrote {len(df)} ratings to {sys.argv[2]}")


if __name__ == "__main__":
    main()

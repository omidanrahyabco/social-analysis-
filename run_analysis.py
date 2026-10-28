"""Run the whole pipeline on one dataset folder.

    python run_analysis.py data/example_10p_2sem --out outputs/example_10p_2sem
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cohort import io, models, network, report, simulate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="folder with the six CSV files")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default outputs/<data name>)")
    ap.add_argument("--runs", type=int, default=1000, help="simulation runs for the forecast")
    ap.add_argument("--steps", type=int, default=2, help="waves to forecast ahead")
    ap.add_argument("--fragment", action="store_true", help="also write report.fragment.html (no <html> wrapper)")
    args = ap.parse_args()
    out = args.out or Path("outputs") / args.data.name

    c = io.load(args.data)
    for issue in c.issues:
        print(f"[{issue['level']}] {issue['file']}: {issue['msg']}")
    print(f"loaded {c.n} people, {c.T} waves, {len(c.dyads)} ratings, {len(c.events)} events")

    waves = network.analyse(c)
    fitted = models.fit(c)
    print(f"tie-change model: R2={fitted['r2']:.2f} on {fitted['n_obs']} pair-transitions")
    one = simulate.one_step(c, fitted)
    free = simulate.backtest(c, fitted)
    fc = simulate.forecast(c, fitted, steps=args.steps, runs=args.runs)

    results = report.build(c, waves, fitted, one, free, fc, args.data.name)
    report.write(results, out)
    if args.fragment:
        report.write(results, out, standalone=False)
    for p in results["predictions"]:
        print(f"  {p['p']:>5.0%}  {p['title']}")
    print(f"wrote {out / 'report.html'} and {out / 'results.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Power-law fits on the shuffled Apollo pilots (2 Nov 2026).

Fits L(D) = E + B * D^-beta (D in billions of tokens) to the smoothed training
cross-entropy of each shuffled run over its constant-learning-rate phase
(steps 1000 to 27465), once with beta free and once with beta fixed at 0.28
(Chinchilla) and 0.40, and converts the prediction to the unseen shard 31 by
the measured held-out minus training offset and the measured anneal bonus.
Also checks the fit by fitting on the first 0.45B tokens and predicting the
end of the constant phase.

Inputs: reports/pilot_results/curves/<run>.csv and
reports/pilot_results/diag/forget_<run>.json. Needs numpy and scipy
(e.g. uv run --with numpy --with scipy python scripts/isambard/fit_scaling.py).
"""
import csv, json, os, sys
import numpy as np
from scipy.optimize import curve_fit

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "pilot_results")
T = 32768
DECAY_START = 27465
RUNS = {
    "0.6B next-token": "pilot_0.6b_medical_apollo_ntp_shuffled",
    "0.6B distilled": "pilot_0.6b_medical_apollo_kd_shuffled",
    "1.7B next-token": "pilot_1.7b_medical_apollo_ntp_shuffled",
    "1.7B distilled": "pilot_1.7b_medical_apollo_kd_shuffled",
}


def power(D, E, B, beta):
    return E + B * D ** (-beta)


def load(run):
    with open(os.path.join(ROOT, "curves", f"{run}.csv")) as handle:
        rows = [(int(r["step"]), float(r["ce"])) for r in csv.DictReader(handle)]
    steps = np.array([r[0] for r in rows]); ce = np.array([r[1] for r in rows])
    held = json.load(open(os.path.join(ROOT, "diag", f"forget_{run}.json")))["loss"]
    return steps, ce, {int(k): v["31"] for k, v in held.items()}


def main():
    out = {}
    for label, run in RUNS.items():
        steps, ce, held = load(run)
        win = 25
        smooth = np.convolve(ce, np.ones(win) / win, "valid"); st = steps[win - 1:]
        final_train = smooth[-1]
        keep = (st >= 1000) & (st <= DECAY_START)
        D = st[keep] * T / 1e9; L = smooth[keep]
        idx = np.unique(np.geomspace(1, len(D) - 1, 400).astype(int))
        D, L = D[idx], L[idx]
        offset = float(np.mean([held[k] - smooth[np.argmin(abs(st - k))] for k in (10000, 16000, 20000, 25000)]))
        fits = {}
        for name, fixed in (("free", None), ("beta=0.28", 0.28), ("beta=0.40", 0.40)):
            if fixed is None:
                p, _ = curve_fit(power, D, L, p0=[1.8, 0.4, 0.4], bounds=([0, 0, 0.05], [4, 50, 2]), maxfev=20000)
                half = D <= 0.45
                ph, _ = curve_fit(power, D[half], L[half], p0=[1.8, 0.4, 0.4], bounds=([0, 0, 0.05], [4, 50, 2]), maxfev=20000)
                check = {"pred_at_0.9B": float(power(D[-1], *ph)), "actual_at_0.9B": float(L[-1])}
            else:
                q, _ = curve_fit(lambda D, E, B: power(D, E, B, fixed), D, L, p0=[1.8, 0.4], bounds=([0, 0], [4, 50]))
                p = [q[0], q[1], fixed]
                half = D <= 0.45
                qh, _ = curve_fit(lambda D, E, B: power(D, E, B, fixed), D[half], L[half], p0=[1.8, 0.4], bounds=([0, 0], [4, 50]))
                check = {"pred_at_0.9B": float(power(D[-1], qh[0], qh[1], fixed)), "actual_at_0.9B": float(L[-1])}
            anneal = float(power(1.0, *p) - final_train)
            fits[name] = {
                "E": float(p[0]), "B": float(p[1]), "beta": float(p[2]),
                "rms": float(np.sqrt(((power(D, *p) - L) ** 2).mean())),
                "anneal_bonus": anneal, "half_fit_check": check,
                "heldout_shard31_pred": {str(x): float(power(x, *p) - anneal + offset) for x in (1, 3, 9, 20, 50)},
                "heldout_shard31_limit": float(p[0] - anneal + offset),
            }
        out[label] = {"run": run, "heldout_minus_train": offset, "heldout_shard31": held, "fits": fits}
        print(f"== {label}: offset {offset:+.3f}, held-out final {held[30517]:.3f}")
        for name, f in fits.items():
            print(f"   {name:9s} E={f['E']:.3f} beta={f['beta']:.3f} rms={f['rms']:.4f} | shard 31 at "
                  + " ".join(f"{k}B:{v:.3f}" for k, v in f["heldout_shard31_pred"].items())
                  + (f" | half-fit check {f['half_fit_check']['pred_at_0.9B']:.3f} vs {f['half_fit_check']['actual_at_0.9B']:.3f}" if f["half_fit_check"] else ""))
    json.dump(out, open(os.path.join(ROOT, "scaling_fit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

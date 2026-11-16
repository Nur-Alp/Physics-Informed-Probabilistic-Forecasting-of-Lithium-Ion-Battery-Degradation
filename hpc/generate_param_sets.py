"""
Generates N parameter-override sets for the HPC degradation sweep.

Each entry in the output JSON is one sub-job's input: a fidelity level
(which mechanisms are switched on) plus a dict of PyBaMM parameter
overrides, sampled with multiplicative log-normal jitter around the
OKane2022 defaults.

The parameters varied here (cracking rate, LAM current-driven rate,
SEI growth rate) are a placeholder starting set, not a finalised
choice. Confirm the actual set and jitter ranges with Greg before
treating sweep results as final: they drive the whole Objective-4
simulator-fidelity comparison (Action plan.md item 17).

Usage:
    python generate_param_sets.py --n 200 --seed 0 --out param_sets.json
"""
import argparse
import json

import numpy as np

FIDELITY_OPTIONS = {
    "sei_only": {
        "SEI": "solvent-diffusion limited",
    },
    "sei_plating": {
        "SEI": "solvent-diffusion limited",
        "lithium plating": "partially reversible",
    },
    "sei_plating_lam": {
        "SEI": "solvent-diffusion limited",
        "lithium plating": "partially reversible",
        "particle mechanics": "swelling only",
        "loss of active material": "stress-driven",
    },
}

# Placeholder jitter spec: {parameter name: multiplicative log-normal sigma}.
# A sigma of 0.3 means roughly a +/-35% spread around the OKane2022 default.
JITTER_SPEC = {
    "Negative electrode cracking rate": 0.3,
    "SEI kinetic rate constant [m.s-1]": 0.3,
    "Negative electrode LAM constant proportional term [s-1]": 0.3,
}


def sample_param_sets(n, seed, fidelity_levels):
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        fidelity = fidelity_levels[i % len(fidelity_levels)]
        overrides = {}
        for name in JITTER_SPEC:
            sigma = JITTER_SPEC[name]
            # multiplicative factor only; applied to the default value
            # at solve time in run_one_curve.py (needs the default looked
            # up from the parameter set, so we store the factor here).
            overrides[name] = float(np.exp(rng.normal(0, sigma)))
        samples.append({"index": i + 1, "fidelity": fidelity, "factors": overrides})
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, help="number of curves")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fidelity",
        nargs="+",
        default=["sei_plating_lam"],
        choices=list(FIDELITY_OPTIONS),
        help="fidelity level(s) to cycle through across the N samples",
    )
    parser.add_argument("--out", default="param_sets.json")
    args = parser.parse_args()

    samples = sample_param_sets(args.n, args.seed, args.fidelity)
    with open(args.out, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"wrote {len(samples)} parameter sets to {args.out}")


if __name__ == "__main__":
    main()

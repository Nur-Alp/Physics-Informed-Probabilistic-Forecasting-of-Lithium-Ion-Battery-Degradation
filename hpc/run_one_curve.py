"""
Runs exactly one degradation curve for the HPC array sweep.

Called as:
    python run_one_curve.py $PBS_ARRAY_INDEX

Reads sample #PBS_ARRAY_INDEX (1-based) from param_sets.json (produced
by generate_param_sets.py), builds the matching PyBaMM model/options,
solves to 80% capacity (or the cycle cap below, whichever comes
first), and writes one compact JSON result file. Designed so each
array sub-job is independent and short enough to fit a few-hour
walltime, per the ICL RCS array-jobs guide.
"""
import json
import sys
import time
from pathlib import Path

import pybamm

CYCLE_CAP = 20000
TERMINATION = "80% capacity"
PROTOCOL = (
    "Discharge at 1C until 2.5 V",
    "Charge at 0.5C until 4.2 V",
    "Hold at 4.2 V until C/50",
)
PARAM_SETS_FILE = "param_sets.json"
RESULTS_DIR = Path("results")


def build_model(fidelity):
    import generate_param_sets as gps

    return pybamm.lithium_ion.SPMe(options=gps.FIDELITY_OPTIONS[fidelity])


def main():
    idx = int(sys.argv[1])
    with open(PARAM_SETS_FILE) as f:
        param_sets = json.load(f)
    sample = next(s for s in param_sets if s["index"] == idx)

    model = build_model(sample["fidelity"])
    parameter_values = pybamm.ParameterValues("OKane2022")
    for name, factor in sample["factors"].items():
        default = parameter_values[name]
        if callable(default):
            parameter_values.update({name: lambda T, d=default, f=factor: f * d(T)})
        else:
            parameter_values.update({name: factor * default})

    experiment = pybamm.Experiment([PROTOCOL] * CYCLE_CAP, termination=TERMINATION)
    sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)

    t0 = time.time()
    try:
        sol = sim.solve(save_at_cycles=50)
        sv = sol.summary_variables
        cap = list(sv["Capacity [A.h]"])
        result = {
            "index": idx,
            "fidelity": sample["fidelity"],
            "factors": sample["factors"],
            "cycles_completed": len(cap),
            "capacity_pct": [100 * c / cap[0] for c in cap],
            "termination": str(sol.termination),
            "wall_time_s": time.time() - t0,
        }
    except Exception as e:
        result = {
            "index": idx,
            "fidelity": sample["fidelity"],
            "factors": sample["factors"],
            "error": str(e),
            "wall_time_s": time.time() - t0,
        }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"curve_{idx:05d}.json"
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

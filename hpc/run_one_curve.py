"""
Runs ONE CHUNK of one degradation curve for the HPC array sweep, then exits.

Called repeatedly (by the PBS wrapper loop in run_sweep.pbs) as:
    python run_one_curve.py $PBS_ARRAY_INDEX

Why chunked, not one long solve: a single sim.solve() over the full
~20000-cycle experiment accumulates solver state for the whole run in
one Python process. A local 16gb/90min HPC test (16 Nov 2026) still
hit the cgroup OOM killer at ~16.7GB after 37-51 min, not finished.
Splitting into CHUNK_CYCLES-sized chunks, each run in its own fresh
`python` process (started by the wrapper loop, not forked), means the
OS fully reclaims memory when each chunk's process exits, bounding
peak memory to one chunk regardless of total cycle count. State is
carried across chunks via two small files per curve:
  - results/_state/curve_<idx>.json: cycles so far, capacity history,
    cap0 (first cycle's capacity, for normalising %), fidelity/factors.
  - results/_state/curve_<idx>.pkl: the PyBaMM Solution from the last
    chunk, saved via Solution.save() / loaded via pybamm.load(), used
    only to seed the next chunk's initial conditions via
    model.set_initial_conditions_from(...). Not the full history.
Once a curve finishes (cap reached or an early-termination event),
this writes results/curve_<idx>.json and removes its state files; the
wrapper loop checks for that file's existence to know when to stop
calling this script again for that index.
"""
import json
import sys
import time
from pathlib import Path

import pybamm

CYCLE_CAP = 20000
CHUNK_CYCLES = 500
THRESHOLD_PCT = 80.0
TERMINATION = "80% capacity"

# 'Throughput capacity [A.h]' is a genuinely large but harmless cumulative
# counter (total Ah ever cycled) that trips PyBaMM's generic divergence
# check (max_y_value=100000) over long enough runs, spamming the error log
# (27MB in one 17000-cycle test, 16-17 Nov 2026) without affecting results.
# Raised, not disabled, so a real divergence would still be caught.
pybamm.settings.max_y_value = 1e7
PROTOCOL = (
    "Discharge at 1C until 2.5 V",
    "Charge at 0.5C until 4.2 V",
    "Hold at 4.2 V until C/50",
)
PARAM_SETS_FILE = "param_sets.json"
RESULTS_DIR = Path("results")
STATE_DIR = RESULTS_DIR / "_state"


def build_model(fidelity):
    import generate_param_sets as gps

    return pybamm.lithium_ion.SPMe(options=gps.FIDELITY_OPTIONS[fidelity])


class ScaledParam:
    """Picklable replacement for a closure: sol.save() pickles the whole
    model, including its parameter functions, and a lambda/closure can't
    be pickled (needed for cross-process checkpointing between chunks)."""

    def __init__(self, default, factor):
        self.default = default
        self.factor = factor

    def __call__(self, T):
        return self.factor * self.default(T)


def build_parameter_values(factors):
    parameter_values = pybamm.ParameterValues("OKane2022")
    for name, factor in factors.items():
        default = parameter_values[name]
        if callable(default):
            parameter_values.update({name: ScaledParam(default, factor)})
        else:
            parameter_values.update({name: factor * default})
    return parameter_values


def load_state(idx):
    state_path = STATE_DIR / f"curve_{idx:05d}.json"
    if not state_path.exists():
        return None
    with open(state_path) as f:
        return json.load(f)


def save_state(idx, state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = STATE_DIR / f"curve_{idx:05d}.json"
    with open(state_path, "w") as f:
        json.dump(state, f)


def checkpoint_path(idx):
    return STATE_DIR / f"curve_{idx:05d}.pkl"


def cleanup_state(idx):
    (STATE_DIR / f"curve_{idx:05d}.json").unlink(missing_ok=True)
    checkpoint_path(idx).unlink(missing_ok=True)


def write_final(idx, result):
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"curve_{idx:05d}.json"
    with open(out_path, "w") as f:
        json.dump(result, f)
    cleanup_state(idx)
    print(f"wrote {out_path}")


def main():
    idx = int(sys.argv[1])

    final_path = RESULTS_DIR / f"curve_{idx:05d}.json"
    if final_path.exists():
        print(f"{final_path} already exists, nothing to do")
        return

    with open(PARAM_SETS_FILE) as f:
        param_sets = json.load(f)
    sample = next(s for s in param_sets if s["index"] == idx)

    state = load_state(idx)
    if state is None:
        state = {
            "cycles_so_far": 0,
            "capacity_pct": [],
            "cap0": None,
            "start_time": time.time(),
        }

    remaining = CYCLE_CAP - state["cycles_so_far"]
    chunk_size = min(CHUNK_CYCLES, remaining)
    if chunk_size <= 0:
        write_final(idx, {
            "index": idx,
            "fidelity": sample["fidelity"],
            "factors": sample["factors"],
            "cycles_completed": state["cycles_so_far"],
            "capacity_pct": state["capacity_pct"],
            "termination": "cycle cap reached",
            "wall_time_s": time.time() - state["start_time"],
        })
        return

    model = build_model(sample["fidelity"])
    parameter_values = build_parameter_values(sample["factors"])

    prev_sol = None
    ckpt = checkpoint_path(idx)
    if ckpt.exists():
        prev_sol = pybamm.load(str(ckpt))
        model = model.set_initial_conditions_from(prev_sol, inplace=False)

    experiment = pybamm.Experiment([PROTOCOL] * chunk_size, termination=TERMINATION)
    sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)

    try:
        sol = sim.solve(save_at_cycles=50)
        sv = sol.summary_variables
        cap = list(sv["Capacity [A.h]"])
        cap0 = state["cap0"] if state["cap0"] is not None else cap[0]
        new_pct = [100 * c / cap0 for c in cap]
        state["capacity_pct"].extend(new_pct)
        state["cycles_so_far"] += len(cap)
        state["cap0"] = cap0

        # The per-chunk experiment's own "80% capacity" termination is
        # relative to THAT CHUNK's first cycle (a fresh Simulation each
        # chunk), not the curve's true first cycle, so it never actually
        # fires here. Check the real, global capacity ourselves instead
        # (found the hard way: a test run sailed past 80% to 17000
        # cycles/68.5% before anyone noticed, 16-17 Nov 2026).
        crossed_at = next(
            (i for i, p in enumerate(state["capacity_pct"]) if p <= THRESHOLD_PCT),
            None,
        )
        if crossed_at is not None:
            state["capacity_pct"] = state["capacity_pct"][: crossed_at + 1]
            state["cycles_so_far"] = len(state["capacity_pct"])
            termination = f"{THRESHOLD_PCT}% capacity (checked globally)"
            done = True
        else:
            termination = str(sol.termination)
            done = (
                state["cycles_so_far"] >= CYCLE_CAP
                or len(cap) < chunk_size  # a non-capacity termination event fired
            )
        if done:
            write_final(idx, {
                "index": idx,
                "fidelity": sample["fidelity"],
                "factors": sample["factors"],
                "cycles_completed": state["cycles_so_far"],
                "capacity_pct": state["capacity_pct"],
                "termination": termination,
                "wall_time_s": time.time() - state["start_time"],
            })
        else:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            sol.save(str(ckpt))
            save_state(idx, state)
            print(f"chunk done: {state['cycles_so_far']} cycles so far, continuing")
    except Exception as e:
        write_final(idx, {
            "index": idx,
            "fidelity": sample["fidelity"],
            "factors": sample["factors"],
            "cycles_completed": state["cycles_so_far"],
            "capacity_pct": state["capacity_pct"],
            "error": str(e),
            "wall_time_s": time.time() - state["start_time"],
        })


if __name__ == "__main__":
    main()

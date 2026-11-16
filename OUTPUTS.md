# What every scratch file in this repo is

All of the `scratch_*.py`, `*.log`, `*_data.json` and `*.png` files in the repo
root are from the Phase 1/3 PyBaMM feasibility and mechanism-debugging work
(2-5 Nov 2026), documented narratively in `Action plan.md` item 14. This file
maps each output to the script and run that produced it, so the pile of
scratch files is navigable without re-reading every script. Grouped by stage.

## Stage 1: basic timing checks (2 Nov)

- **`scratch_pybamm_timing_test.py`** - first feasibility check (Action plan
  item 6): SPMe + SEI only, 50 cycles, OKane2022. No saved output file, result
  was read from console at the time.
- **`scratch_pybamm_timing_test_full.py`** - same but with the full mechanism
  stack (SEI + irreversible plating + cracking + SEI-on-cracks + LAM), to
  check worst-case per-cycle cost. No saved output file.
- **`scratch_pybamm_timing_test_v2.py`** - corrected version with an explicit
  80%-capacity termination and a faster-charging protocol. Takes a mechanism
  set name as a CLI arg (`"SEI-only"` or `"full-stack"`). This is the script
  that found the problem: `"full-stack"` with `lithium plating: "irreversible"`
  collapsed to 80% within ~13 cycles, an unphysical result that kicked off the
  whole mechanism investigation. No saved output file.

## Stage 2: isolating which mechanism caused the collapse (2 Nov)

- **`scratch_isolate_mechanism.py`** - re-runs the full stack with each
  mechanism combination enabled one at a time, to find which one(s) actually
  drive the fast collapse. No saved output file.
- **`scratch_tune_lam.py`** - scanned the LAM rate constant across 3 orders of
  magnitude; found it made no difference (still 80% at cycle ~14), ruling out
  LAM's rate constant as the cause. No saved output file.
- **`scratch_plot_fidelity_comparison.py`** → **`fidelity_comparison.png`**.
  Plots SEI-only (slow, realistic fade) against the full stack (collapses in
  ~14 cycles) side by side, under identical OKane2022/1C-0.5C conditions. This
  is the plot that visualises the actual bug: it was `lithium plating:
  "irreversible"`, not a LAM or SEI problem (confirmed by switching to
  `"partially reversible"`, which fixed it).

## Stage 3: chasing a 70%/80% capacity target, repeatedly hitting a solver wall (3 Nov)

All four of these use the corrected `"partially reversible"` plating, and all
four hit the same `IDA_ERR_FAIL` solver failure (O'Kane et al.'s own
porosity-approaches-zero limitation, confirmed against the PyBaMM community
forum) at under 5% capacity fade, well short of the 70-80% target:

- **`scratch_degradation_curve_70pct.py`** → `degradation_curve_70pct.log`.
  Mild 1C/0.5C protocol, cycle cap 10,000. Crashed at cycle 2357 (96.8%
  capacity, wall time 389.8s). Its own `degradation_curve_70pct_data.json` is
  essentially empty (11 bytes): the script's `json.dump` itself crashed on a
  `numpy.int64` serialisation error, after the solve had already failed. The
  real data from this exact run survives instead as **`spme_quick_demo_6000_
  data.json`** (cycle 2357, 96.796%, matches exactly), almost certainly
  re-saved via a quick inline fix at the time rather than by this script.
- **`scratch_degradation_curve_70pct_v2.py`** → `degradation_curve_70pct_v2.
  log` / `degradation_curve_70pct_v2_data.json`. More aggressive 2C/1C
  protocol, cycle cap 4000. Crashed at cycle 1245 (97.4%, 173.5s), same
  failure mode, confirming it wasn't the protocol's fault.
- **`scratch_degradation_curve_80pct_literature.py`** →
  `degradation_curve_80pct_literature.log` / `..._literature_data.json`.
  Tried literature-sourced tuned prefactors (10x dead-lithium decay, 2x SEI
  solvent diffusivity, sourced from a PyBaMM Discourse thread hitting the same
  known failure). Crashed at cycle 1270 (95.1%, 102.4s). Confirms the
  porosity-zero limitation holds even with tuned rates.
- **`scratch_degradation_curve_80pct_dfn.py`** - tried the full DFN model
  instead of SPMe (matching O'Kane et al.'s own notebook), to see if a
  higher-fidelity model avoids the failure. No output beyond the 2-line
  `degradation_curve_80pct_dfn.log`: this run was killed by the OS for memory
  (exit 137) before producing any cycles, because DFN's default full-state
  storage explodes in memory over thousands of cycles, the same class of
  problem later re-discovered and properly fixed for the HPC sweep (see
  `hpc/run_one_curve.py`'s chunking, 16 Nov 2026).
- **`scratch_degradation_curve_80pct_finemesh.py`** - tried O'Kane et al.'s
  own finer particle mesh (r_n=r_p=30) and explicit `IDAKLUSolver`, to check
  whether the failure was a discretisation artefact rather than a true
  physical limit. No output file survives; this run did not complete.

Three comparison plots from this stage:
- **`degradation_curves_comparison.png`** - the two 70% protocol attempts
  (1C/0.5C crashed at 2357, 2C/1C crashed at 1245) plotted together, both
  stopping at the same `IDA_ERR_FAIL`, nowhere near 70%.
- **`mild_protocol_partial_curve.png`** - just the 1C/0.5C attempt's partial
  curve on its own (same data as above), titled with the crash point.
- **`degradation_curves_80pct_attempt_final.png`** - all three 80%-target
  attempts (default 1C/0.5C, default 2C/1C, tuned-prefactor 1C/0.5C) plotted
  together, annotated against O'Kane et al. (2022) Fig. 5a's own published
  result (their full-stack model, 1000 cycles, only reaches ~90-94%), showing
  this is a known limitation of the mechanism stack itself, not a mistake in
  this setup.

## Stage 4: the actual fix (3-5 Nov)

- **`scratch_degradation_fixed_config.py`** - first fix: drop `"SEI on
  cracks"` to `"false"` (O'Kane et al. Section 5.3 identify this exact
  interaction as what clogs negative-electrode porosity). Reached cycle 5444
  (92.9% capacity) with a clean physical stop, not a crash. The matching data
  is **`spme_nocracks_6000_data.json`** (cycle 5444, 92.943%, exact match);
  the script itself only prints to console, this file was saved separately.
  Still short of 80%, since the crack-length-exceeds-radius physical limit
  replaced the solver crash as the new stopping point.
- **`spm_vs_spme.png`** / **`spm_quick_demo_6000_data.json`** - the same
  fixed config run on the simpler SPM model instead of SPMe, for comparison
  (reached 5051 cycles/93.4% in 77s, faster but similar shape). No `.py`
  source survives for this one, it's referenced only in a comment in
  `scratch_degradation_fixed_config.py`.
- **`spme_fix_comparison.png`** - before/after plot: with `"SEI on cracks":
  "true"` (crashes at 2357) vs. `"false"` (clean run to 5444), titled
  "Dropping SEI-on-cracks more than doubles the usable run."
- **`scratch_degradation_swelling_only.py`** - the final, fully-working
  config: drops the crack-growth submodel entirely (`"particle mechanics":
  "swelling only"`), removing the crack-length state that was itself hitting
  its physical limit in the previous fix. This is the configuration
  documented as the production setting in `Action plan.md` item 14, and later
  reused (with a custom cracking submodel instead, see `PyBaMM-dev/`) for the
  HPC sweep in `hpc/`.
  - **`spme_swellingonly_6000_data.json`** - an earlier check of this config
    capped at 6000 cycles (didn't terminate early, reached 92.0% at cycle
    6000, cap reached rather than 80%).
  - **`spme_swellingonly_18000_data.json`** - the real, final verification:
    cycle cap raised to 20,000, reaches a genuine 80.00% capacity at cycle
    15,760, no crash. This is the headline result.
  - **`real_80pct_degradation_curve.png`** - the plot of that exact
    15,760-cycle run, titled "SPMe, full degradation stack (swelling only),
    OKane2022 defaults."

## Not from any saved script

- **`spm_quick_demo_data.json`** (1000 cycles, 98.17%) - an early, short
  exploratory check. No matching script or plot survives; likely a one-off
  inline run before any of these scripts were saved.

## Where this connects to later work

- The memory-growth problem rediscovered on CX3 (16 Nov 2026, see `hpc/
  README.md`) is the same class of issue first hit here with DFN
  (`scratch_degradation_curve_80pct_dfn.py`): PyBaMM accumulating full solver
  state over thousands of cycles in one process. The HPC fix (chunking the
  solve across separate processes) is the general-purpose version of that
  same lesson.
- The `"swelling only"` production config here is the *built-in* PyBaMM
  cracking submodel being switched off entirely. The crack-saturation model
  built in `PyBaMM-dev/` (16 Nov 2026, from Dharshannan's write-up) instead
  keeps cracking on but fixes its unbounded growth, a different, complementary
  approach to the same underlying particle-mechanics problem.

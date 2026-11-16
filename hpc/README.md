# Running the degradation sweep on CX3

One-time setup and submission steps for generating N simulated degradation
curves on Imperial's CX3 HPC, confirmed against the ICL RCS user guide
(16 Nov 2026) except where marked as inference.

## 1. Connect (you do this yourself, needs your password + Unified Access)

```bash
ssh your_username@login.cx3.hpc.imperial.ac.uk
```

Must be on the Imperial network or connected to Unified Access first.
Password auth only, no SSH keys on the login nodes.

## 2. One-time environment setup, on a login node

```bash
module load miniforge/3
miniforge-setup
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda config --set auto_activate_base false
conda config --remove channels defaults
conda config --add channels conda-forge
conda create -n fyp python=3.12
conda activate fyp
pip install pybamm numpy
```

The `pip install pybamm` step is inference, not something the RCS conda
page documents (it only covers `conda install` from conda-forge). pip
works fine inside an activated conda env; this is standard practice, just
not RCS-specific confirmed text.

## 3. Get the code onto the cluster

From your own machine:
```bash
git clone https://github.com/Nur-Alp/Physics-Informed-Probabilistic-Forecasting-of-Lithium-Ion-Battery-Degradation.git
```
on the login node (or `git pull` if already cloned), then `cd` into `hpc/`.

## 4. Generate the parameter sets

```bash
python generate_param_sets.py --n 200 --seed 0 --fidelity sei_plating_lam --out param_sets.json
```

Check `generate_param_sets.py`'s `JITTER_SPEC` and `FIDELITY_OPTIONS` before
a real run, the varied parameters and jitter ranges are placeholders pending
Greg's input (Action plan.md item 17).

## 5. Submit the array job

```bash
qsub run_sweep.pbs
qstat -u $USER      # S column: Q queued, R running, H held, B array
qdel <jobid>        # cancel if needed
```

`run_sweep.pbs` requests 1 CPU / 4GB / 3h per curve and runs 200 curves at
up to 50 concurrently (`-J 1-200%50`). The walltime is sized from a local
timing check (0.092 s/cycle for this exact config, measured 16 Nov 2026)
scaled to the 20000-cycle cap, with a safety margin for a slower/shared
core. Check actual usage after the first few sub-jobs finish and adjust.
The `%50` concurrency cap is a starting guess, not a confirmed per-user
limit, check with RCS/Greg if you want to push it higher.

## 6. Where results land

Each sub-job writes `results/curve_<index>.json`. This is a small amount
of data (a few KB each), fine to keep in `$HOME` for now. If the sweep
grows much larger, move to RDS project space instead of `$EPHEMERAL`
(ephemeral auto-deletes after 30 days and isn't backed up, wrong for
data you want to keep).

## Open items

- Whether you have an RDS project space allocated is still unconfirmed,
  worth checking before a large-scale run.
- Per-user concurrent-job limits on CX3 aren't confirmed from the docs,
  only the array's own `%` cap is under your control.

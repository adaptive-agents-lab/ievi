# Reproducing the experiments

Run the commands from the repository root. Use a new output directory for each independent run.

## Install

Use Python 3.13 and the pinned CPU dependencies. Figure generation also requires Poppler's `pdftoppm` executable on `PATH`, installed through your system package manager.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproduction.txt
.venv/bin/python -m pip check
```

On Windows, create the environment with `python -m venv .venv` and use `.venv/Scripts/python` in place of `.venv/bin/python`. A short checkout path, such as `C:\ievi`, avoids path-length limits in deeply nested experiment outputs.

## Verify the data

```bash
.venv/bin/python scripts/verify_reference_inventory.py
.venv/bin/python reproduce.py verify-inputs all
.venv/bin/python reproduce.py verify-dataset
.venv/bin/python -m studies.cliffwalk verify
```

These commands check input inventories, file hashes, result archives, and the Cliffwalk study's source and data manifest. The result archives are unpacked automatically when needed; `reproduce.py materialize-dataset` can unpack them in advance.

## Generate figures and tables

```bash
.venv/bin/python reproduce.py render-paper --output-root artifacts/paper
```

This command generates the 19 figure assets and 11 empirical tables from the supplied numerical results. It writes figures, tables and input metadata to the chosen output directory. It does not run the learners or planning solvers.

## Run the Cliffwalk learning comparison

```bash
.venv/bin/python -m studies.cliffwalk select --output runs/cliffwalk-selection
.venv/bin/python -m studies.cliffwalk run --output runs/cliffwalk --methods q os ie --workers 3
.venv/bin/python -m studies.cliffwalk compare --output runs/cliffwalk
```

`select` recomputes the hyperparameter ranking from the supplied tuning results and checks the selected settings. `run` executes the three selected methods on the 60 held-out seeds. `compare` checks the resulting numerical records against the supplied data.

The Cliffwalk comparison reports exact equality separately from agreement within its portability tolerance of `1e-5` relative and absolute error.

To recompute the plotted means and intervals from the supplied held-out trajectories:

```bash
.venv/bin/python -m studies.cliffwalk render-data --output runs/cliffwalk-statistics
```

See [the methods](docs/experiments.md) for tuning, sample accounting, and uncertainty estimates.

## Run planning and policy-evaluation experiments

```bash
.venv/bin/python reproduce.py list
.venv/bin/python reproduce.py run all --output-root runs/experiments
.venv/bin/python reproduce.py status all --output-root runs/experiments
.venv/bin/python reproduce.py compare all --output-root runs/experiments
```

The full suite is computationally expensive. `list` shows the available task IDs and categories; replace `all` with an ID or category to run a subset. `--resume` reuses only completed attempts whose commands, inputs, code, numerical environment and output hashes still match. Every new attempt writes its command, environment, logs and output inventory to a separate directory.

Run timing tasks separately after other computational jobs have finished. Timing values depend on hardware and load; comparison checks convergence and work counts while reporting timing differences.

Discrete outcomes, seeds, grids, policies and accounting counts require exact agreement. Continuous numerical comparisons use relative tolerance `1e-9` and absolute tolerance `1e-11` unless the task declares otherwise. A completed execution and a successful numerical comparison are reported separately.

## Check the installation

```bash
.venv/bin/python -m pytest tests -q
```

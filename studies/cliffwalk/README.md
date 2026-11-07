# Cliffwalk

This directory reproduces Figure 1(b): Q-learning, OS-Dyna and IE-Dyna on Cliffwalk
with support-uniform model corruption at lambda 0.8 and gamma 0.99. The reported
metric is the all-state policy-value L1 error,
`100 * sum(abs(Vpi - Vstar)) / sum(abs(Vstar))`.

Run from the repository root in the reproduction environment:

```console
python -m studies.cliffwalk verify
python -m studies.cliffwalk select --output runs/cliffwalk-selection
python -m studies.cliffwalk run --output runs/cliffwalk --methods q os ie --workers 3
python -m studies.cliffwalk render-data --output runs/cliffwalk-statistics
python -m studies.cliffwalk compare --output runs/cliffwalk
```

Tuning uses seeds 20–29. All 44 source files and all 36 learning-rate schedules
that can affect these three selections are included: 36 Q-learning candidates,
108 OS-Dyna candidates (B=10, 30, 100), and 1,440 IE-Dyna candidates (J=1, 3;
B=10, 30; h=0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30).
Q-learning is model-free, so its source file uses lambda 0.

Each method minimizes the normalized trapezoidal area of the tuning-seed mean
error through 40,000 nominal samples. Candidate order and first-minimum tie
handling match the source selector. Nonfinite areas receive positive infinity;
no candidate is dropped. The reported scope contains no nonfinite candidate
areas. `data/tuning-ranking.csv` records every score, rank and selected flag;
`data/tuning-candidate-scores.npz` contains the same complete 1,584 candidates.
The 44 tuning NPZ files retain their complete recorded fields and checksums;
selection reads only `pol|all|L1` from those source records.

The selected settings are Q-learning `v:n^-0.8`, B=10; OS-Dyna `c0.01`, B=30;
and IE-Dyna `v:3/(3+n)`, J=1, B=10, h=0.03. Their exact parameters and tuning
areas are in `data/results/cliffwalk/selection.json`. No held-out observation
changes a setting.

Held-out evaluation uses all seeds 250–309, every checkpoint from 0 through
80,000 nominal samples at intervals of 2,000, and the method's last complete
update within each budget. OS-Dyna completes 79,980 actual samples at the final
checkpoint. Each bundled held-out file contains only the selected configuration,
reported metric, seeds and actual sample counts. The source replay batch groups
are retained in `execution-groups.json` to preserve numerical batch dimensions;
the runner exports only the reported configuration and metric. The replay
generator uses 400,000 steps, CPU JAX and 64-bit arithmetic.

`load_panel_data()` returns errors `(3,41,60)`, means `(3,41)`, intervals `(2,3,41)`,
nominal samples, actual samples, seeds and methods in Q-learning, OS-Dyna,
IE-Dyna order. Pointwise 95% percentile bootstrap bands use 20,000 whole-seed
trajectory resamples and RNG seed 20261106. Statistics are calculated before the
`log10(1 + error_percent)` display transform. All 41 checkpoints and 60 held-out
seeds are included in the figure data.

`comparison.json` reports exact equality and maximum absolute error across all
41-by-60 reported observations. It separately checks relative and absolute
portability tolerances of 1e-5; the command fails if they are exceeded.
`manifest.json` records the packaged inputs and implementation checksums.

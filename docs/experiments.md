# Experiment methods

The configuration files and result manifests specify the complete parameter grids, random seeds and output files. The paper's empirical evaluation covers planning, policy evaluation, surrogate-solve accuracy, horizon scaling, runtime, and reinforcement learning.

## Planning and policy evaluation

The experiments use tabular Cliffwalk, Frozen Lake, Maze and Four Rooms environments, with support-uniform or self-loop perturbations of the transition model. VI and OS-VI are compared with fixed-step and adaptive implicit-Euler methods. Configuration-specific discount factors, perturbation grids, query budgets and step schedules are supplied with each task.

The main planning panel uses discount `gamma=0.99` and a strict policy-error threshold of `1e-3`. A stable hit is the first recorded query after which all remaining recorded errors stay strictly below the threshold; a first hit is the first recorded error below it. No-hit cases remain in the data. The main panel uses the adaptive carry-forward method. Separate configurations evaluate restarting and restoring the best-residual iterate when the step size decreases. The latter also preserves the accepted policy after a rejected candidate.

Practical adaptive plots count charged outer attempts, including rejected attempts. Their initialization, cached calculations and batched inactive-lane operations are outside that counter. The controlled solve-accuracy study instead counts actual full operator applications, including initialization. These axes use different accounting conventions.

Practical policy evaluation uses float64 direct linear solves. Practical Control uses warm-started surrogate value iteration with a successive-iterate stopping rule; tolerances and iteration caps are specified by each configuration.

The controlled solve-accuracy comparison uses the same surrogate value-iteration solver for OS-VI and IE-VI, including warm starts, sweep caps, tie breaking and certified solve accuracy. Its stopping condition is

```text
||T_hat U - U||_infinity <= (1 - beta) * delta_solve.
```

The two solve tolerances are `1e-8` and `1e-2`, evaluated over 19 corruption settings. This is a deterministic fixed-instance comparison, with no seed uncertainty bars. True and surrogate operator applications are counted at their call sites. Weighted cost is `N_P + kappa * N_Phat`, where `kappa` is the surrogate-to-true operator cost ratio. Offline policy evaluation is diagnostic and is excluded from algorithm cost.

## Horizon scaling and runtime

The horizon experiment uses a fixed policy in Modified Cliffwalk, 30 paired replications, and `H` in `10, 20, 40, 80, 160, 320, 640`, with `gamma=1-1/H`. The surrogate sample schedules are `n(H)=80` and `n(H)=16H`. VI, OS-VI and the conservative and refined IE-VI schedules use the same paired empirical models. Recorded quantities include stable-hit query counts, true and surrogate applications, factorizations, linear solves and solver time.

Horizon plots show medians and interquartile ranges. Finite-range log-log slope intervals use 4,000 paired-seed bootstrap resamples, reported over the full horizon grid and over `H>=80`. Each replication has five nested timing measurements; these measurements do not add independent surrogate seeds.

The runtime benchmark uses one fixed 10,000-state simulator instance, single-threaded CPU execution, `gamma=0.995`, and ten randomized-order repetitions of all five methods. No timing outliers are discarded. Reported mean intervals use two-sided Student-t 95% intervals; paired speedups are calculated within each repetition. These repetitions measure timing variability on one problem instance.

Results report solver time and operator preparation plus solver time separately. Preparation includes coefficient construction and kernel warm-up or compilation encountered. Both views exclude reference-solution construction, calibration, imports, process startup and file I/O. Required factorization is included in the horizon experiment's algorithm time, while surrogate sampling and experiment preparation are excluded. Runtime comparisons should be interpreted together with convergence, residuals and measured work; elapsed times are machine dependent.

## Cliffwalk learning comparison

The environment is stochastic 6-by-6 Cliffwalk with `gamma=0.99`. State-action pairs are sampled uniformly from a generative replay tape. OS-Dyna and IE-Dyna mix each visited empirical transition row with the uniform distribution over its observed successor support, using weights `0.2` and `0.8`. Unvisited rows are uniform over all states. Q-Learning has no surrogate corruption parameter.

Each method is tuned on ten seeds, `20–29`, using mean integrated error through 40,000 nominal samples. The selected settings are then evaluated on 60 held-out seeds, `250–309`. Tuning and evaluation seeds do not overlap.

The learning-rate grid contains 36 constant, per-visit and global-index schedules. OS-Dyna batch sizes are `10, 30, 100`. IE-Dyna batch sizes are `10, 30`, with one or three planning rounds and fixed steps from `0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30`. Surrogate planning uses five policy-iteration sweeps per round.

| Method | Selected learning rate | Batch size | Planning rounds | Fixed step |
| --- | --- | ---: | ---: | ---: |
| Q-Learning | per-visit `n^-0.8` | 10 | 1 | — |
| OS-Dyna | constant `0.01` | 30 | 1 | — |
| IE-Dyna | per-visit `3/(3+n)` | 10 | 1 | `0.03` |

Every recorded policy is evaluated on the true MDP. Percentage policy error is

```text
100 * sum_s |V_policy(s) - V_optimal(s)| / sum_s |V_optimal(s)|.
```

Reference values use tabular policy iteration; policy values use linear-system evaluation. Evaluation is recorded every 2,000 nominal samples from zero through 80,000, using the last completed update within each budget. OS-Dyna's final complete batch is at 79,980 actual samples. The endpoint therefore denotes the final evaluation at the nominal budget.

The plotted curves are means over the 60 held-out seeds. Bands are 95% pointwise percentile bootstrap intervals, using 20,000 resamples of whole seed trajectories and bootstrap seed `20261106`. The same resampled seed indices are used for all methods. Means and intervals are calculated before applying the plot's logarithmic transformation of one plus percentage error.

Integrated error uses trapezoidal integration on the nominal sample grid. Q-Learning has lower integrated error over the first 40,000 samples. Fixed-step IE-Dyna has lower late-stage error. The final mean percentage errors round to `0.07%` for IE-Dyna, `2.1%` for Q-Learning and `5.1%` for OS-Dyna. These findings apply to this environment and protocol.

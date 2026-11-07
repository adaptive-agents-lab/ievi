# Data and outputs

The repository supplies the numerical results used to generate the paper's 19 figure assets and 11 empirical tables, together with experiment configurations, solver sources and result inventories. The main empirical figure combines a planning comparison with the Cliffwalk learning comparison described in [the methods](experiments.md).

| Location | Contents |
| --- | --- |
| `configs/` | Task definitions and experiment parameter grids |
| `datasets/reference/planning/` | Planning and policy-evaluation result archives and file inventories |
| `studies/cliffwalk/` | Reinforcement-learning sources, tuning results and held-out trajectories |
| `paper/manuscript/` | Accompanying paper and identity metadata |

The 73 planning and policy-evaluation archives supply figure inputs, empirical table values and the numerical checks reported in Appendix N. Each archive lists its files and their SHA256 hashes. The residual-incumbent results include the 38 trajectories used in Table N.11.

The [figure and table index](coverage/coverage.json) lists the source tasks for each numbered figure and empirical table.

For Figure 1(b), the Cliffwalk study supplies all 1,584 tuning candidates for Q-learning, OS-Dyna and IE-Dyna at the reported condition. The held-out data contain the three selected settings, with every checkpoint from all 60 test seeds. The plotted quantity is normalized all-state policy-value L1 error.

`reproduce.py verify-dataset` checks result-archive hashes and every listed member. Experiment manifests identify input files, source files and expected outputs. New runs record their commands, numerical environments, logs and output hashes. Generated figures and tables include the identities of their numerical inputs.

## Coverage

Most figures are calculated from trajectories or numerical summaries. Some appendix control and schedule-sensitivity plots use supplied per-configuration hit records; their raw trajectories are not included, so rendering these plots does not recompute hits from trajectories. The schedule-sensitivity tables are calculated from the supplied hit and aggregate records.

Deterministic target values for applicable planning reruns are calculated from the specified MDP and checked using Bellman and policy-evaluation residuals. Four Rooms configurations with an environment variant construct the specified seed-zero variant and check its MDP fingerprint before execution.

The two-state figure is an analytic illustration. Its script evaluates the specified two-state linear model and includes the illustrative VI cone construction; it is separate from the benchmark measurements. Environment schematics are generated from their geometry specifications.

Numerical comparisons check the supplied observations and declared invariants. Rendering a figure, verifying its input hashes and independently executing a solver are distinct operations. PDF bytes can differ across plotting environments; numerical values, labels and geometry provide the relevant comparison.

The accompanying paper is a fixed reference document. Rendering writes new assets to the requested output directory.

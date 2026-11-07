"""Render planning and reinforcement-learning panels from measured data."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

VI_COLOR = '#9D9D9D'
OS_COLOR = '#1696D2'
IE_COLOR = '#CA5800'
TEXT_COLOR = '#242424'
DISPLAY_SEGMENTS = [(0., .6, .30), (.6, .8, .45), (.8, 1., .25)]


def lambda_to_display(lam):
    cumulative = 0.
    for start, end, width in DISPLAY_SEGMENTS:
        if lam <= end + 1e-9:
            return cumulative + width * max(0., min(1., (lam - start) / (end - start)))
        cumulative += width
    return cumulative


def decorate(ax):
    ax.grid(True, which='major', axis='y', color='#DADADA', alpha=.72, linewidth=.42)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_color(TEXT_COLOR)
        ax.spines[side].set_linewidth(.62)
    ax.tick_params(axis='both', which='major', labelsize=7., length=2.4, width=.55)


def planning_panel(ax, metadata):
    rows = metadata['rows']
    budget = metadata['run_budget']
    ax.set_yscale('log')
    ax.set_xlim(-.025, 1.025)
    ax.set_ylim(.7, budget * 2.1)
    ax.set_yticks([1, 10, 100, 1000], labels=[r'$1$', r'$10$', r'$10^2$', r'$10^3$'])
    ticks = [0., .2, .4, .6, .7, .8, .9, 1.]
    ax.set_xticks([lambda_to_display(x) for x in ticks], labels=[f'{x:g}' for x in ticks])
    ax.set_xlabel(r'$\lambda$ (piecewise)', labelpad=2., fontsize=8.)
    ax.set_ylabel(r'stable-hit query $q_{\rm SH}(10^{-3})$', labelpad=2., fontsize=8.)
    for boundary in (.6, .8):
        ax.axvline(lambda_to_display(boundary), color='#DADADA', ls=':', lw=.55)
    vi_hits = {row['vi_stable_hit'] for row in rows}
    if len(vi_hits) != 1 or None in vi_hits:
        raise ValueError('Planning VI baseline must have one finite lambda-independent hit')
    vi_hit = vi_hits.pop()
    ax.axhline(vi_hit, color=VI_COLOR, ls=(0, (1., 1.8)), lw=1.05)
    ax.text(.025, vi_hit * 1.16, rf'VI $q^\star={vi_hit}$', color=VI_COLOR, fontsize=7.)
    for key, color, ls, marker in [('osvi_stable_hit', OS_COLOR, (0, (3., 1.6)), 's'),
                                    ('aievi_stable_hit', IE_COLOR, '-', 'o')]:
        # None creates a line break, so a missing hit is never interpolated.
        xs = [lambda_to_display(row['lambda']) for row in rows]
        ys = [float(row[key]) if row[key] is not None else math.nan for row in rows]
        ax.plot(xs, ys, color=color, ls=ls, lw=1.12, marker=marker, ms=3.0,
                mfc='white' if marker == 's' else color, mec=color, mew=.72)
        missing = [x for x, row in zip(xs, rows, strict=True) if row[key] is None]
        ax.plot(missing, [budget * 1.65] * len(missing), ls='None', marker='^',
                color=color, mfc='white', mec=color, mew=.72, ms=4.)
    decorate(ax)


def planning_metadata(rows, run_budget=1000, source_tasks=None):
    """Describe measured stable-hit rows used by Figure 1(a)."""
    return {'run_budget': run_budget, 'rows': rows,
            'source_tasks': source_tasks or ['selected-control-cliffwalk-gamma099', 'planning-05_threshold'],
            'regime': 'one-round carry-forward Control', 'gamma': .99,
            'environment': '6x6 Cliffwalk', 'error_mode': 'support_uniform', 'threshold': .001,
            'stable_hit_definition': 'first recorded index after the last nonfinite or >= threshold policy-value error',
            'counting_convention': 'reported active-attempt index; excludes initialization, includes rejected attempts; reference construction and diagnostic policy evaluation excluded',
            'schedule': {'h_init': 1e6, 'h_min': 1., 'beta': .1, 'S': 3},
            'inner_solver': {'type': 'warm-start surrogate value iteration', 'tolerance': 1e-10, 'iteration_cap': 5000},
            'execution_mode': 'accepted practical harness --primary (primary carry-forward; no incumbent restoration)'}


def load_planning_metadata(dataset=None):
    """Read only Figure 1(a)'s verified measured planning streams."""
    from scripts.reference_dataset import DEFAULT_DATASET, resolve_attempt
    from scripts.render_planning import read, record
    dataset = Path(dataset) if dataset else DEFAULT_DATASET
    base = resolve_attempt(dataset, 'selected-control-cliffwalk-gamma099')[0].parent / 'study'
    adaptive = resolve_attempt(dataset, 'planning-05_threshold')[0].parent / 'study'
    base_path, = (base / 'solver/raw').glob('*_traces.json')
    adaptive_path = adaptive / 'solver/raw-traces.json'
    baselines = {float(row['lambda']): row for row in read(base_path)}
    merged = []
    for measured in read(adaptive_path):
        row = dict(measured)
        baseline = baselines[float(row['lambda'])]
        if row.get('seed') != baseline.get('seed'):
            raise ValueError('Planning baseline/adaptive seed mismatch')
        row['traces'] = {**baseline['traces'], **row['traces']}
        merged.append(row)
    # Use the same stable-hit calculation as the appendix renderer, without
    # writing a second assembled trace or rendering any other experiment.
    from scripts.render_planning import hit
    rows = []
    for measured in merged:
        row = {'lambda': measured['lambda'], 'seed': measured.get('seed')}
        for public, key in [('vi', 'vi'), ('osvi', 'osvi'), ('aievi', 'waterfall')]:
            trace = measured['traces'][key]
            q, errors = trace['queries'], trace.get('errors', trace.get('policy_value_errors'))
            row.update({f'{public}_stable_hit': hit(q, errors), f'{public}_first_hit': hit(q, errors, stable=False),
                        f'{public}_last_query': q[-1], f'{public}_last_error': errors[-1], f'{public}_max_error': max(errors)})
        rows.append(row)
    metadata = planning_metadata(rows)
    metadata['source_files'] = [record(base_path), record(adaptive_path)]
    return metadata


def draw_planning_figure(path, metadata):
    """Draw the native left panel only, with the published geometry and styling."""
    width, height = 198., 183.6
    fig = plt.figure(figsize=(width / 72., height / 72.), dpi=280)
    ax = fig.add_axes([40 / width, 34 / height, 147 / width, 98 / height])
    planning_panel(ax, metadata)
    fig.text(40 / width, 172 / height, '(a) Planning', fontsize=8., va='baseline')
    handles = [
        Line2D([], [], color=VI_COLOR, ls=(0, (1., 1.8)), lw=1.05, label='VI'),
        Line2D([], [], color=OS_COLOR, ls=(0, (3., 1.6)), lw=1.10, marker='s',
               mfc='white', mec=OS_COLOR, ms=3., label='OS-VI'),
        Line2D([], [], color=IE_COLOR, ls='-', lw=1.15, marker='o',
               mfc=IE_COLOR, mec=IE_COLOR, ms=3., label='Adaptive IE-VI')]
    options = dict(loc='center', ncol=3, frameon=False, fontsize=7.,
                   handlelength=1., columnspacing=.6, handletextpad=.35,
                   borderpad=0., borderaxespad=0.)
    fig.legend(handles=handles, bbox_to_anchor=(113.5 / width, 158.5 / height), **options)
    fig.legend(handles=[Line2D([], [], color='#5F5F5F', ls='None', marker='^',
                              mfc='white', mec='#5F5F5F', ms=4., label='no stable hit')],
               bbox_to_anchor=(113.5 / width, 143.5 / height), **{**options, 'ncol': 1})
    fig.savefig(path, bbox_inches=None)
    plt.close(fig)


def cliffwalk_statistics(panel):
    """Recompute the reference pointwise bands, resampling whole seed trajectories."""
    import numpy as np
    errors = np.asarray(panel['errors'], dtype=float)
    methods = list(panel['methods'])
    times, seeds = np.asarray(panel['nominal_samples']), np.asarray(panel['seeds'])
    if methods != ['q', 'os', 'ie'] or errors.shape != (3, 41, 60):
        raise ValueError('Figure panel requires the three audited 41-by-60 raw curves')
    if not np.array_equal(times, np.arange(0, 80001, 2000)) or not np.array_equal(seeds, np.arange(250, 310)):
        raise ValueError('Figure panel requires nominal samples 0:2000:80000 and held-out seeds 250:309')
    if not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError('Policy-value errors must be finite and nonnegative')
    means = errors.mean(axis=-1)
    rng = np.random.default_rng(20261106)
    indices = rng.integers(0, 60, size=(20000, 60))
    weights = np.zeros((20000, 60), dtype=np.int16)
    np.add.at(weights, (np.repeat(np.arange(20000), 60), indices.ravel()), 1)
    intervals = np.empty((2, 3, 41))
    for i in range(3):
        bounds = np.quantile(errors[i] @ weights.T / 60, [.025, .975], axis=1)
        constant = np.ptp(errors[i], axis=1) == 0
        bounds[:, constant] = means[i, constant]
        intervals[:, i] = bounds
    if not np.array_equal(means, panel['means']) or not np.array_equal(intervals, panel['intervals']):
        raise ValueError('Raw recomputation differs from the reference mean/bootstrap arrays')
    return means, intervals


def render_figure(output_stem: Path, planning=None, data_root=None, dataset=None):
    """Render Figure 1 and verify its raw statistics and 300-dpi reference image."""
    import shutil
    import subprocess
    import tempfile
    import numpy as np
    from PIL import Image
    from pypdf import PdfReader, PdfWriter
    from scripts.reproduction_manifest import guard_output_path, sha256, write_json
    from studies.cliffwalk.study import load_panel_data

    root = Path(__file__).resolve().parents[1]
    data_root = Path(data_root) if data_root else root / 'studies/cliffwalk/data'
    output_stem = Path(output_stem).resolve()
    guard_output_path(output_stem.parent, root)
    if output_stem.is_relative_to(data_root.resolve()):
        raise ValueError('Figure-panel outputs must not enter the packaged input data')
    outputs = [output_stem.with_suffix(s) for s in ('.pdf', '.png', '.metadata.json', '.data.npz')]
    if any(path.exists() for path in outputs):
        raise ValueError('Choose a new output stem; existing figure-panel artifacts are preserved')
    rasterizer = shutil.which('pdftoppm')
    if rasterizer is None:
        raise ValueError('Figure verification requires Poppler pdftoppm on PATH')
    reference = data_root / 'figure-reference.pdf'
    reference_sha = '5c1b971767630201761af3a1ab02bd134ed4b28be8e0b5e8aaec72ee83718a86'
    if sha256(reference) != reference_sha:
        raise ValueError('Figure 1 reference hash mismatch')
    planning = planning if planning is not None else load_planning_metadata(dataset)
    panel = load_panel_data(data_root=data_root)
    means, intervals = cliffwalk_statistics(panel)
    times = np.asarray(panel['nominal_samples'])
    methods = list(panel['methods'])
    colors = {'q': '#8B8B8B', 'os': OS_COLOR, 'ie': IE_COLOR}
    labels = {'q': 'Q-learning', 'os': 'OS-Dyna', 'ie': 'IE-Dyna'}
    styles = {'q': (0, (1, 1.8)), 'os': (0, (3, 1.6)), 'ie': '-'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
                         'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42,
                         'text.color': TEXT_COLOR, 'axes.labelcolor': TEXT_COLOR,
                         'xtick.color': TEXT_COLOR, 'ytick.color': TEXT_COLOR,
                         'figure.constrained_layout.use': False, 'figure.autolayout': False})
    width, height = 396, 183.6
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cliffwalk-render-') as temporary:
        temporary = Path(temporary)
        fig = plt.figure(figsize=(width / 72, height / 72))
        ax = fig.add_axes([235 / width, 34 / height, 147 / width, 98 / height])
        for method in ('q', 'os', 'ie'):
            i = methods.index(method)
            ax.fill_between(times, np.log10(1 + intervals[0, i]), np.log10(1 + intervals[1, i]),
                            color=colors[method], alpha=.16, lw=0)
            ax.plot(times, np.log10(1 + means[i]), color=colors[method], ls=styles[method],
                    lw=1.12, zorder=2 + (method == 'ie'))
        ax.set_xlim(-800, 80800)
        ax.set_ylim(-.04, np.log10(121))
        ax.set_xticks([0, 40000, 80000], labels=['0', '40k', '80k'])
        ax.set_yticks(np.log10(1 + np.array([0, 1, 10, 100])), labels=['0', '1', '10', '100'])
        ax.set_xlabel(r'environment samples $t$', fontsize=8, labelpad=2)
        ax.set_ylabel(r'Policy-value $L_1$ error (%)', fontsize=8, labelpad=2)
        decorate(ax)
        ax.set_axisbelow(True)
        fig.text(235 / width, 172 / height, r'(b) RL · $\gamma=0.99$, $\lambda=0.8$', fontsize=8)
        handles = [Line2D([], [], color=colors[m], ls=styles[m], lw=1.12, label=labels[m]) for m in ('q', 'os', 'ie')]
        fig.legend(handles=handles, loc='center', ncol=3, frameon=False, fontsize=7,
                   handlelength=1.1, columnspacing=.9, handletextpad=.4, labelspacing=.45,
                   bbox_to_anchor=(308.5 / width, 152 / height))
        overlay = temporary / 'overlay.pdf'
        fig.savefig(overlay)
        plt.close(fig)
        planning_pdf = temporary / 'planning.pdf'
        draw_planning_figure(planning_pdf, planning)
        writer = PdfWriter()
        page = writer.add_blank_page(width=width, height=height)
        page.merge_page(PdfReader(overlay).pages[0])
        page.merge_page(PdfReader(planning_pdf).pages[0])
        candidate = temporary / 'candidate.pdf'
        with candidate.open('wb') as stream:
            writer.write(stream)
        for source, name in [(candidate, 'candidate'), (reference, 'reference')]:
            subprocess.run([rasterizer, '-singlefile', '-r', '300', '-png', str(source), str(temporary / name)],
                           check=True, capture_output=True)
        actual, expected = [np.asarray(Image.open(temporary / (name + '.png'))) for name in ('candidate', 'reference')]
        if not np.array_equal(actual, expected):
            raise ValueError('Figure 1 differs from the 300-dpi reference; no candidate exported')
        shutil.copy2(candidate, outputs[0])
        shutil.copy2(temporary / 'candidate.png', outputs[1])
    np.savez_compressed(outputs[3], errors=panel['errors'], means=means, intervals=intervals,
                        nominal_samples=times, seeds=panel['seeds'], methods=np.array(methods))
    metadata = {'figure': 'main_empirical_results_two_panel', 'main_panel': 'paper',
                'panel_count': 2, 'size_inches': [5.5, 2.55], 'displayed_methods': ['q', 'os', 'ie'],
                'raw_data_methods': methods, 'metric': 'pol|all|L1',
                'normalization': '100 * sum_s |Vpi(s)-Vstar(s)| / sum_s |Vstar(s)|',
                'gamma': .99, 'lambda': .8, 'family': 'support-uniform', 'selection': 'per_lam',
                'n_bootstrap': 20000, 'bootstrap_rng_seed': 20261106, 'bootstrap_unit': 'whole seed trajectory',
                'tuning_seeds': list(range(20, 30)), 'test_seeds': np.asarray(panel['seeds']).tolist(),
                'nominal_samples': times.tolist(),
                'interval': '95% pointwise percentile bootstrap', 'statistics_before_log_transform': True,
                'display_transform': 'log10(1 + policy_value_L1_error_percent)',
                'raw_statistics_match_reference_arrays': True,
                'reference_sha256': reference_sha, 'pixel_identical_to_reference_at_300dpi': True,
                'planning_panel_width_points': 198,
                'planning': planning, 'planning_native_content_only': True,
                'source_data': panel['source_data'],
                'outputs': {p.name: sha256(p) for p in (outputs[0], outputs[1], outputs[3])}}
    write_json(outputs[2], metadata)
    return metadata

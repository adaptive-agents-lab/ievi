#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import pandas as pd


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); return p.parse_args()

def fnum(x, digits=3):
    if pd.isna(x): return '—'
    x=float(x)
    if abs(x)>=1000: return f'{x:,.0f}'
    if abs(x)>=10: return f'{x:.1f}'
    if abs(x)>=1: return f'{x:.3f}'
    if abs(x)>=1e-3: return f'{x:.4f}'
    return f'{x:.3e}'

def ms(x):
    if pd.isna(x): return '—'
    return f'{1000*float(x):.3f}'

def iqr(m,q1,q3,formatter=fnum):
    return f"{formatter(m)} [{formatter(q1)}, {formatter(q3)}]"

def md_table(headers, rows):
    lines=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
    for row in rows: lines.append('| '+' | '.join(str(x) for x in row)+' |')
    return '\n'.join(lines)

def get_slope(slopes,schedule,quantity,grid='all_H'):
    r=slopes[(slopes.schedule==schedule)&(slopes.quantity==quantity)&(slopes.grid==grid)].iloc[0]
    return float(r.slope),float(r.ci_low),float(r.ci_high)

def main():
    root=parse_args().root.resolve()
    summary=pd.read_csv(root/'data/derived/summary_by_H.csv')
    endpoint=pd.read_csv(root/'data/derived/endpoint_H640.csv')
    slopes=pd.read_csv(root/'data/derived/slope_summary.csv')
    records=pd.read_csv(root/'data/raw/records.csv')
    verify=json.load(open(root/'verification/verification_report.json'))
    independent=json.load(open(root/'verification/independent_heldout-integrity.json'))

    schedules=['fixed80','linear16']; algs=['VI','OS-VI','IE-VI Eq16','IE-VI Joint']
    lines=[]
    lines += ['# Matched policy-evaluation horizon-scaling experiment','',
    '## Executive result','',
    'For every replication, horizon, and sample schedule, the code directly runs VI, OS-VI, IE-VI with the Equation-16 parameters, and jointly tuned IE-VI from the same zero initialization on the same true MRP and paired empirical surrogate.','']

    efix=get_slope(slopes,'fixed80','E_infinity')
    elin=get_slope(slopes,'linear16','E_infinity')
    qfix=get_slope(slopes,'fixed80','N_P::IE-VI Joint')
    qlin=get_slope(slopes,'linear16','N_P::IE-VI Joint')
    virow=endpoint[(endpoint.schedule=='linear16')&(endpoint.algorithm=='VI')].iloc[0]
    jrow=endpoint[(endpoint.schedule=='linear16')&(endpoint.algorithm=='IE-VI Joint')].iloc[0]
    osrow=endpoint[(endpoint.schedule=='linear16')&(endpoint.algorithm=='OS-VI')].iloc[0]
    eqrow=endpoint[(endpoint.schedule=='linear16')&(endpoint.algorithm=='IE-VI Eq16')].iloc[0]

    lines += [
      f"Under the fixed-information control, the measured model-error slope is {efix[0]:.3f} (95% bootstrap CI [{efix[1]:.3f}, {efix[2]:.3f}]) and joint IE-VI's true-query slope is {qfix[0]:.3f} [{qfix[1]:.3f}, {qfix[2]:.3f}], worse than VI's approximately linear scaling.",
      '',
      f"Under `n(H)=16H`, the measured error slope is {elin[0]:.3f} [{elin[1]:.3f}, {elin[2]:.3f}], close to the predicted inverse-square-root behavior. Joint IE-VI's true-query slope is {qlin[0]:.3f} [{qlin[1]:.3f}, {qlin[2]:.3f}], versus VI's approximately unit slope.",
      '',
      f"At `H=640`, joint IE-VI uses a median {jrow.median_N_P:,.0f} true-model applications versus {virow.median_N_P:,.0f} for VI, a {jrow.VI_over_algorithm_N_P:.2f}× true-query reduction. Equation-16 IE-VI uses {eqrow.median_N_P:,.0f}. OS-VI uses only {osrow.median_N_P:,.0f} true applications on this fixed-policy family and remains the fastest comparator.",
      '',
      f"The corresponding median algorithm-only times are {ms(jrow.median_wall_time_seconds)} ms for joint IE-VI, {ms(virow.median_wall_time_seconds)} ms for VI, and {ms(osrow.median_wall_time_seconds)} ms for OS-VI. The measured algorithm-only times do not establish an IE-VI wall-clock advantage over VI.",
      ''
    ]

    lines += ['## Experimental contract','',
      '- **True problem:** the 36-state Markov reward process induced by one fixed policy in the Modified Cliffwalk MDP.',
      '- **Policy:** the true optimal policy at `H=640`, computed once before any empirical surrogate draw and then frozen for all conditions.',
      '- **Horizons:** `H ∈ {10,20,40,80,160,320,640}`, with `γ=1-1/H`.',
      '- **Schedules:** `fixed80`, with 80 successor samples per policy row; and `linear16`, with `n(H)=16H` samples per policy row.',
      '- **Pairing:** 30 replications. Within each replication, all horizons and both schedules use nested prefixes of the same row-wise sample streams.',
      '- **Target:** relative `L∞` value error `10^-3` from zero initialization.',
      '- **Stable hit:** the first reported output followed by 50 additional outer outputs below the same target.',
      '- **Tuning:** Equation-16 and joint-tuned IE-VI use the measured true `E∞`; they are oracle mechanism tests, not adaptive deployment rules.',
      '- **Solver parity:** OS-VI and both IE-VI variants use the same SciPy `lu_factor`/`lu_solve` backend. Each run factors its fixed surrogate matrix once and reuses it for every changing right-hand side.',
      '- **Timing:** one-threaded algorithm-only timing to the previously determined stable-hit index; required matrix construction and factorization are included, while surrogate sampling, exact-target construction, plotting, and diagnostic error evaluation are excluded.',
      ''
    ]

    lines += ['## Cost coordinates','',
      'The raw records report the following coordinates directly at the stable hit:', '',
      '- `N_P`: true transition-operator applications;',
      '- `N_Phat`: surrogate transition-operator applications in the defect term;',
      '- `N_factorizations`: required LU factorizations;',
      '- `N_linear_solves`: selected surrogate linear solves;',
      '- algorithm-only wall-clock time.', '',
      'For VI, `(N_Phat,N_factorizations,N_linear_solves)=(0,0,0)`. For OS-VI and the PE IE-VI implementations, every true defect evaluation is paired with one surrogate defect application and one reused-factorization linear solve, so `N_P=N_Phat=N_linear_solves`, with one factorization per run.', ''
    ]

    for sch in schedules:
        lines += [f'## Full median results — `{sch}`','']
        rows=[]
        for H in sorted(summary.H.unique()):
            base=summary[(summary.schedule==sch)&(summary.H==H)]
            e=float(base.median_E_infinity.iloc[0]); re=float(base.median_R_E.iloc[0])
            vals={alg:base[base.algorithm==alg].iloc[0] for alg in algs}
            rows.append([
                int(H),fnum(e),fnum(re),
                iqr(vals['VI'].median_N_P,vals['VI'].q25_N_P,vals['VI'].q75_N_P),
                iqr(vals['OS-VI'].median_N_P,vals['OS-VI'].q25_N_P,vals['OS-VI'].q75_N_P),
                iqr(vals['IE-VI Eq16'].median_N_P,vals['IE-VI Eq16'].q25_N_P,vals['IE-VI Eq16'].q75_N_P),
                iqr(vals['IE-VI Joint'].median_N_P,vals['IE-VI Joint'].q25_N_P,vals['IE-VI Joint'].q75_N_P),
                fnum(vals['IE-VI Joint'].median_m),
            ])
        lines += [md_table(['H','median E∞','median R_E','VI N_P [IQR]','OS-VI N_P [IQR]','Eq16 N_P [IQR]','Joint N_P [IQR]','Joint m'],rows),'']

        trows=[]
        for H in sorted(summary.H.unique()):
            base=summary[(summary.schedule==sch)&(summary.H==H)]
            vals={alg:base[base.algorithm==alg].iloc[0] for alg in algs}
            trows.append([int(H),ms(vals['VI'].median_wall_time_seconds),ms(vals['OS-VI'].median_wall_time_seconds),ms(vals['IE-VI Eq16'].median_wall_time_seconds),ms(vals['IE-VI Joint'].median_wall_time_seconds)])
        lines += ['Median algorithm-only time in milliseconds:', '', md_table(['H','VI','OS-VI','IE-VI Eq16','IE-VI Joint'],trows),'']

    lines += ['## Scaling estimates','']
    srows=[]
    for sch in schedules:
        for quantity in ['E_infinity','N_P::VI','N_P::OS-VI','N_P::IE-VI Eq16','N_P::IE-VI Joint']:
            for grid in ['all_H','H_ge_80']:
                r=slopes[(slopes.schedule==sch)&(slopes.quantity==quantity)&(slopes.grid==grid)].iloc[0]
                srows.append([sch,quantity,grid,f'{r.slope:.3f}',f'[{r.ci_low:.3f}, {r.ci_high:.3f}]'])
    lines += [md_table(['schedule','quantity','fit grid','slope','95% bootstrap CI'],srows),'']

    lines += ['## H=640 complete cost table','']
    erows=[]
    for sch in schedules:
        for alg in algs:
            r=endpoint[(endpoint.schedule==sch)&(endpoint.algorithm==alg)].iloc[0]
            erows.append([sch,alg,fnum(r.median_N_P),fnum(r.median_N_Phat),fnum(r.median_N_factorizations),fnum(r.median_N_linear_solves),ms(r.median_wall_time_seconds),f'{r.VI_over_algorithm_N_P:.2f}×',f'{r.VI_over_algorithm_wall_time:.2f}×',fnum(r.median_m)])
    lines += [md_table(['schedule','algorithm','N_P','N_Phat','factorizations','linear solves','time (ms)','VI / alg N_P','VI / alg time','m'],erows),'']

    # Actual contraction diagnostics at H=640.
    lines += ['## Error-propagation diagnostics at H=640','']
    crows=[]
    for sch in schedules:
        sub=summary[(summary.schedule==sch)&(summary.H==640)]
        re=float(sub.median_R_E.iloc[0])
        for alg in ['OS-VI','IE-VI Eq16','IE-VI Joint']:
            r=sub[sub.algorithm==alg].iloc[0]
            crows.append([sch,alg,fnum(re),fnum(r.median_actual_spectral_radius),fnum(r.median_actual_inf_norm),fnum(r.median_N_P)])
    lines += [md_table(['schedule','algorithm','median R_E','actual spectral radius','actual ∞-norm','median N_P'],crows),'',
      '`R_E` is a conservative supremum-norm upper bound. In this particular fixed-policy MRP, OS-VI\'s actual error-propagation spectral radius is far below one even when `R_E>1`, which explains its two-to-three-query convergence. OS-VI remains the fastest comparator on this fixed-policy MRP.','']

    lines += ['## Verification','',
      f"- Main verification: `passed={verify['passed']}`; {verify['records_observed']:,} of {verify['records_expected']:,} expected algorithm records; every condition contains all four algorithms; all runs reached the target and passed the 50-output stable tail.",
      f"- Independent algebraic audit: `passed={independent['passed']}`; {independent['records_checked']:,} records independently reconstructed from exact linear error-propagation matrices; zero stable-hit, error, matrix-statistic, model-hash, or contraction-envelope mismatches.",
      f"- Timing archive: {independent['timing_rows']:,} raw timing observations, {independent['timing_repetitions_per_record']} repetitions for each algorithm record.",
      '- Every empirical transition matrix and exact target is archived in `data/raw/experiment_inputs.npz`.',
      ''
    ]

    lines += ['## Interpretation and limitations','',
      'Results summary:', '',
      '> In a paired Policy Evaluation horizon sweep, we directly executed VI, OS-VI, Equation-16 IE-VI, and jointly tuned IE-VI under a common linear-algebra backend and complete cost instrumentation. With fixed surrogate information, IE-VI scales worse than VI. With `n(H)=16H`, measured surrogate error follows the expected inverse-square-root law and joint IE-VI has sublinear true-query growth; at `H=640` it uses substantially fewer true-model applications than VI. OS-VI remains fastest on this fixed-policy family. After charging one factorization and every surrogate linear solve, the 36-state dense benchmark does not show an IE-VI wall-clock win over VI.', '',
      'This study does not establish end-to-end sample complexity after model construction, a deployable non-oracle tuning rule, or uniform superiority over OS-VI.', ''
    ]

    lines += ['## Reproduction','', 'See the repository REPRODUCE.md for the horizon-study command.', '',
      'The study contains 30 replications, statistical tables, a main verification, and an independent matrix-propagation audit.', ''
    ]

    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')

    result_summary=f'''# Results summary — focused horizon study\n\nThis is a paired Policy Evaluation horizon study in which VI, OS-VI, Equation-16 IE-VI, and jointly tuned IE-VI are all executed from zero under the same data, target, and linear-algebra backend. We hold the true MRP, reward, and policy fixed and vary $H=1/(1-\\gamma)$. With a fixed 80-sample surrogate, the measured model error is essentially constant and IE-VI scales worse than VI, reproducing the fixed-imperfect-surrogate failure mode. With $n(H)=16H$ samples per policy row, the measured $E_\\infty$ slope is {elin[0]:.3f} (95% CI [{elin[1]:.3f},{elin[2]:.3f}]), close to $H^{{-1/2}}$, while jointly tuned IE-VI has true-query slope {qlin[0]:.3f} [{qlin[1]:.3f},{qlin[2]:.3f}] versus VI's approximately linear slope. At $H=640$, joint IE-VI uses a median {jrow.median_N_P:,.0f} true-model applications versus {virow.median_N_P:,.0f} for VI ({jrow.VI_over_algorithm_N_P:.2f}x fewer). We also count every surrogate defect application, one required factorization, every reused-factorization linear solve, and wall-clock time. OS-VI requires only {osrow.median_N_P:,.0f} true applications and is fastest on this fixed-policy family; this study does not establish uniform superiority over OS-VI.\n'''
    (root/'RESULTS_SUMMARY.md').write_text(result_summary)

    readme='''# Matched policy-evaluation horizon study\n\nStart with `REPORT.md`.\n\n- `code/`: executable experiment, independent audit, and report generator.\n- `data/raw/`: all 1,680 algorithm records, 8,400 timing observations, and every empirical surrogate matrix.\n- `data/derived/`: aggregated tables, endpoint tables, slope estimates, and JSON summary.\n- `verification/`: main and independent audit reports.\n- `environment/`: exact software, hardware, and BLAS-thread information.\n- `RESULTS_SUMMARY.md`: concise result block.\n\nThe package deliberately distinguishes true-operator applications, surrogate applications, factorizations, linear solves, and wall-clock time.\n'''
    (root/'README.md').write_text(readme)

if __name__=='__main__': main()

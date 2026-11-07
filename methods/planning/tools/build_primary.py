"""Build scientific summaries from retained measurements; never run solvers or edit release metadata."""
from pathlib import Path
import json, argparse, sys, subprocess, re
import numpy as np
import pandas as pd
from scipy.stats import t
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input-root',type=Path,required=True)
p.add_argument('--output-root',type=Path,required=True)
args=p.parse_args(); F=args.input_root.resolve(); B=args.output_root.resolve(); D=B/'data'
D.mkdir(parents=True,exist_ok=True); (B/'figures').mkdir(parents=True,exist_ok=True)
if B == F or F in B.parents:
    raise SystemExit('Derived output must be outside retained results/benchmarks; use a separate output directory.')
def write(p,s):
    (B/p).write_text(s,encoding='utf-8')
def table(df):
    d=df.copy().replace({'IE-VI Joint':'IE-VI (refined schedule)','IE-VI Eq16':'IE-VI (conservative schedule)'}).fillna('NA')
    def v(x): return f'{x:.6g}' if isinstance(x,float) else str(x).replace('|','\\|')
    return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(v(x) for x in row)+' |' for row in d.itertuples(index=False,name=None))+'\n'
def summarize(df,keys,metrics):
    rows=[]
    for key,g in df.groupby(keys):
        if not isinstance(key,tuple): key=(key,)
        for m in metrics:
            x=g[m].dropna(); n=len(x); sd=x.std(ddof=1); se=sd/np.sqrt(n); margin=t.ppf(.975,n-1)*se
            rows.append(dict(zip(keys,key),metric=m,n=n,mean=x.mean(),sd=sd,se=se,ci95_low=x.mean()-margin,ci95_high=x.mean()+margin,median=x.median(),q25=x.quantile(.25),q75=x.quantile(.75),minimum=x.min(),maximum=x.max()))
    return pd.DataFrame(rows)
m1=pd.read_csv(F/'control_accuracy/raw/control_accuracy_summary.csv')
m2=pd.read_csv(F/'horizon_scaling/data/raw/records.csv')
ms=pd.read_csv(F/'horizon_scaling/data/derived/summary_by_H.csv')
slopes=pd.read_csv(F/'horizon_scaling/data/derived/slope_summary.csv')
w=pd.read_csv(F/'solver-timing-measurements.csv')
original_w=pd.read_csv(F/'wallclock/data/raw/benchmark_runs.csv')
assert np.allclose(w.solver_seconds,original_w.solver_seconds)
assert np.allclose(w.runtime_seconds,w.solver_seconds)
recorded_sum=w.solver_seconds+w.true_setup_seconds+w.surrogate_setup_seconds
assert np.allclose(w.setup_plus_solver_seconds,recorded_sum,rtol=0,atol=1e-10)
# Primary accounting uses solver time; preserve the separately recorded fields.
w['setup_plus_solver_seconds']=w['solver_seconds']+w['true_setup_seconds']+w['surrogate_setup_seconds']
w['runtime_seconds']=w['solver_seconds']
# Retained measurements are read-only; write new derivatives only.
assert len(m1)==95 and len(m2)==1680 and len(w)==50
assert m2.seed.nunique()==30 and m2.groupby(['schedule','H','algorithm']).size().eq(30).all()
assert m2.hit.all() and (m2.error_after_stable_tail<=.001*(1+1e-10)).all()
assert w.converged.all() and (w.final_relative_true_residual<=1e-5).all()
assert w.groupby('method').size().eq(10).all()
assert w.loc[w.method!='VI','inner_solver_converged'].all()
assert np.allclose(w.setup_plus_solver_seconds,w.solver_seconds+w.true_setup_seconds+w.surrogate_setup_seconds)
assert json.loads((F/'horizon_scaling/verification/verification_report.json').read_text())['passed']
hs=summarize(m2,['schedule','H','algorithm'],['N_P','N_Phat','E_infinity','wall_time_median_seconds'])
ws=summarize(w,['method'],['runtime_seconds','solver_seconds','true_setup_seconds','surrogate_setup_seconds','true_operator_calls','surrogate_operator_calls','final_relative_true_residual'])
hs.to_csv(D/'horizon-statistics.csv',index=False); ws.to_csv(D/'runtime-statistics.csv',index=False)
methods=['VI','IEVI_VI_Ks4','IEVI_VI_Ks8','IEVI_VI_Ks16','IEVI_VI_Ks32']
wt=ws[ws.metric=='runtime_seconds'].set_index('method').loc[methods].reset_index()
paired=w.pivot(index='repetition',columns='method',values='runtime_seconds')
ratios=pd.DataFrame({'repetition':paired.index})
for method in methods[1:]: ratios[method]=paired.VI.values/paired[method].values
ratios.to_csv(D/'paired-speedups.csv',index=False)
speed=summarize(ratios.melt(id_vars='repetition',var_name='method',value_name='speedup'),['method'],['speedup'])
speed.to_csv(D/'speedup-statistics.csv',index=False)
ie=m1[(m1.algorithm=='IEVI_Waterfall_common_VI')&(m1.delta_solve==.01)].sort_values('lambda')
tight=m1[(m1.algorithm=='IEVI_Waterfall_common_VI')&(m1.delta_solve==1e-8)].sort_values('lambda')
assert np.array_equal(ie.stable_N_P.values,tight.stable_N_P.values)
reduction=float(np.median(tight.stable_N_Phat.values/ie.stable_N_Phat.values))
cols=['algorithm','lambda','delta_solve','stable_N_P','stable_N_Phat','kappa_star_vs_vi']
hardware=json.loads((F/'hardware-and-protocol.json').read_text())
protocol='''# Final experimental protocol and hardware

The measurements are in `results/benchmarks/`. Execution dates and the measured software environment are recorded in hardware-and-protocol.json.

## Computer and execution

AMD Ryzen AI Max+ 395, one processor socket, 16 physical CPU cores, 32 logical processors; 128 GiB installed memory (134,912,741,376 bytes available to Windows as total physical memory). Windows 11 Pro, version 10.0.26200. CPU execution only; no GPU acceleration. Each numerical process uses one numerical-library thread. The deterministic corruption sweep uses up to four processes; horizon and timing studies execute sequentially. We launched no competing experiment during timing; ordinary desktop/background activity was not eliminated and no exclusive CPU affinity was imposed. Hardware was measured through Windows CIM.

## Sampling and uncertainty

- E1/E2: deterministic Modified Cliffwalk control benchmark, all 19 corruption settings and both tolerances, each computed once. No statistical seed or uncertainty bar applies. This is a fixed-instance study, not an estimate over random environments. Missing stable hits remain missing, never replaced by zeros.
- E3: 30 independent empirical-surrogate seeds, 202609170000 through 202609170029, seven horizons, two sampling schedules, four methods (1,680 solver outcomes). Seeds are paired across algorithms and horizons, using nested surrogate samples. Each seed has five repeated algorithm timing measurements; these are nested measurements, not 150 independent seeds. The 420 model arrays and seed IDs are included.
- E4: one fixed 10,000-state simulator instance (simulator seed 20260729), ten randomized-order timing repetitions, order seeds 21260729 through 21260738. These are ten timing repetitions, not ten independent problem instances. Runtime repetitions do not measure environmental generalization. All five methods are run in every repetition. No outliers are discarded.
- Summary CSVs include n, mean, sample standard deviation, standard error, two-sided Student-t 95% confidence interval for the mean, median, quartiles, and range. The t intervals are descriptive under an independence approximation; desktop timing noise may be correlated. Horizon figures retain median and interquartile bands, explicitly labelled. Horizon slope intervals use 4,000 paired-seed bootstrap resamples of log-log fits to median counts, with both the full range and H >= 80 reported. These are finite-range descriptive slopes, not asymptotic proofs.

## Runtime scope and reproducibility

E4 reports solver-only time as the primary comparison and recorded operator-preparation plus solver time as a separate comparison. Solver timing covers iterations and convergence checks; reference-solution error assessment is outside it. The second comparison adds true_setup_seconds and, for IE-VI, the selected surrogate_setup_seconds to solver_seconds. Preparation includes coefficient construction, kernel warm-up and any JIT cache loading or compilation encountered. Simulator objects are prepared before randomized method execution. Both views exclude initial vector and threshold preparation, reward generation, reference-solution construction, pilot calibration, imports, process startup and file I/O. Neither view is end-to-end experiment-generation time. The raw fields remain separate and unchanged.

The raw benchmark CSVs and solver-timing-measurements.csv retain runtime_seconds equal to solver_seconds. Primary plots and statistics use solver_seconds. The additional table summarizes setup_plus_solver_seconds; each row is verified against the sum of its three recorded components. Speedups are paired within repetition before averaging. The additional accounting is derived from the recorded solver and operator-preparation measurements.

E3 algorithm timings include required matrix factorization; surrogate sampling and experiment preparation are excluded. E1/E2 wall_seconds_full_run is diagnostic full-budget duration, not time to first/stable hit and not a headline runtime comparison.

Reproduction: install `benchmarks/` in a fresh Python environment with the versions below, then run `python tools/run_primary.py --output-root new-solver-run`. It executes all three jobs sequentially with pinned numerical thread counts and refuses nonempty output directories. Original measurement files are protected from overwrite. Regenerate summaries and plots using `python tools/build_primary.py --input-root new-solver-run --output-root new-derived` after all runs finish. Derived products are written outside the retained inputs, under `generated/` by default. Rendering retained results requires no solver run. The repository REPRODUCE.md gives commands for the main-paper figures. SHA-256 hashes bind the packaged sources and outputs.
'''
protocol+='\nSoftware captured at run time:\n\n'+json.dumps({k:hardware[k] for k in ['python','packages']},indent=2)+'\n'
hardware_description=(f"CPU: {hardware.get('cpu')}; physical cores: {hardware.get('physical_cores')}; "
    f"logical processors: {hardware.get('logical_processors')}; installed RAM GiB: {hardware.get('installed_ram_GiB')}; "
    f"OS: {hardware.get('os')}. CPU execution only; no GPU acceleration. "
    "Each numerical process uses one numerical-library thread. The deterministic corruption sweep uses up to four processes; horizon and timing studies execute sequentially. No exclusive CPU affinity is imposed. Unmeasured hardware fields remain null in the recorded metadata.")
protocol=re.sub(r'AMD Ryzen AI Max.*?computer nickname\.',lambda match:hardware_description,protocol,count=1,flags=re.S)
write('EXPERIMENT-PROTOCOL.md',protocol)
sl=slopes[slopes.grid=='all_H']
ep=ms[ms.H==640][['schedule','algorithm','median_N_P','q25_N_P','q75_N_P','median_E_infinity']]
runtime_table=wt[['method','n','mean','sd','se','ci95_low','ci95_high','median','q25','q75']]
evidence='# Experimental evidence\n\nThese summaries are calculated from the supplied numerical measurements. Full hardware, sampling and timing definitions are in EXPERIMENT-PROTOCOL.md.\n\n'
evidence+='## Hardware and statistical unit\n\n'+hardware_description+' E1/E2 are deterministic; E3 uses 30 independent surrogate seeds; E4 uses ten timing repetitions on one fixed simulator, rather than independent environment seeds.\n\n'
evidence+='## E1/E2: weighted cost and tolerance\n\nWeighted cost is N_P + kappa N_Phat. All 19 IE-VI true-query counts agree between tolerances; median surrogate-work reduction is '+f'{reduction:.6g}x.'+' OS-VI has ten successful settings and nine missing stable hits at each tolerance. Blank/NA stable-hit values mean failure within the observation budget.\n\n'+table(m1[cols])
evidence+='\n## E3: horizon scaling\n\n30 paired seeds per condition. Both schedules and all four algorithms are reported. Sublinear finite-range growth does not establish polylogarithmic growth or uniform superiority over OS-VI.\n\n### Endpoint at H=640 (medians and quartiles)\n\n'+table(ep)+'\n### Log-log slopes and paired-seed bootstrap 95% intervals\n\n'+table(sl)+'\n### Mean, standard deviation and standard error of true queries\n\n'+table(hs[hs.metric=='N_P'][['schedule','H','algorithm','n','mean','sd','se','ci95_low','ci95_high']])
evidence+='\n## E4: solver-only runtime (seconds)\n\nTen randomized-order repetitions of the same fixed simulator; no discarded samples. The primary comparisons below exclude setup; the following separate comparison includes the recorded operator-preparation costs.\n\n'+table(runtime_table)+'\n### Paired speedups (VI time / IE-VI time)\n\n'+table(speed[['method','n','mean','sd','se','ci95_low','ci95_high','median']])
evidence+='\n### Counts and residuals\n\n'+table(w.groupby('method',as_index=False).agg(true_calls=('true_operator_calls','median'),surrogate_calls=('surrogate_operator_calls','median'),maximum_residual=('final_relative_true_residual','max')))
evidence+='\n## Numerical data and plotting map\n\n- E1/E2: results/benchmarks/control_accuracy/raw/control_accuracy_summary.csv; trajectories and rejected/accepted attempts are also retained.\n- E3: results/benchmarks/horizon_scaling/data/raw/records.csv (one row per seed/condition/method); timing_repetitions.csv; archived surrogate arrays; horizon-statistics.csv and horizon_scaling/data/derived/slope_summary.csv.\n- E4: results/benchmarks/solver-timing-measurements.csv (all 50 solver-only measurements and order seeds); runtime-statistics.csv; paired-speedups.csv; speedup-statistics.csv.\n- Raw result paths above are package-relative. Regenerated statistics are in data/ beside this document; supplied statistics are in results/benchmarks/.\n- tools/build_primary.py regenerates this evidence, both timing summaries, and four figures without modifying raw inputs. CSVs retain full precision; displayed tables are rounded for reading.\n- These measurements use the variant without incumbent restoration and do not establish a restart-free theorem.\n'
# Secondary, separately labelled accounting from the same 50 preserved rows.
setup_paired=w.pivot(index='repetition',columns='method',values='setup_plus_solver_seconds')
setup_rows=[]
for method in methods:
    x=w.loc[w.method==method,'setup_plus_solver_seconds']; n=len(x); se=x.std(ddof=1)/np.sqrt(n)
    row=dict(method=method,n=n,mean=x.mean(),sd=x.std(ddof=1),se=se,
             ci95_low=x.mean()-t.ppf(.975,n-1)*se,ci95_high=x.mean()+t.ppf(.975,n-1)*se,
             paired_mean_speedup=np.nan,paired_se=np.nan,paired_ci95_low=np.nan,paired_ci95_high=np.nan)
    if method!='VI':
        ratio=setup_paired.VI/setup_paired[method]; rse=ratio.std(ddof=1)/np.sqrt(len(ratio))
        row.update(paired_mean_speedup=ratio.mean(),paired_se=rse,
                   paired_ci95_low=ratio.mean()-t.ppf(.975,len(ratio)-1)*rse,
                   paired_ci95_high=ratio.mean()+t.ppf(.975,len(ratio)-1)*rse)
    setup_rows.append(row)
setup=pd.DataFrame(setup_rows)
setup.to_csv(D/'operator-setup-plus-solver-statistics.csv',index=False)
evidence+='\n## E4: recorded operator-preparation plus solver time\n\nThe same ten repetitions are charged their recorded true-operator preparation and, for IE-VI, the selected surrogate preparation. Preparation includes coefficient construction and kernel warm-up/JIT encountered before randomized solver execution. Other preparation and experiment-generation costs are excluded; this is not an independently timed end-to-end or isolated cold-start run. Speedups are paired within repetition before averaging. Intervals are descriptive Student-t 95% intervals with nine degrees of freedom.\n\n'+table(setup)
evidence+='\n## Regenerated products\n\nCSV summaries, including operator-setup-plus-solver-statistics.csv, are under data/ beside this generated document. Raw measurements are in results/benchmarks/. Figure captions and labels use refined/conservative schedule terminology; raw algorithm IDs are unchanged.\n'
write('EXPERIMENT-EVIDENCE.md',evidence)
subprocess.run([sys.executable,str(R/'tools/render_primary.py'),'--data-root',str(F),'--output-dir',str(B/'figures')],check=True)
checks=dict(control_accuracy_rows=len(m1),horizon_rows=len(m2),timing_rows=len(w),
            setup_sum_verified=True,raw_inputs_written=False,solver_experiments_run=False,
            output_root=B.relative_to(R).as_posix() if B.is_relative_to(R) else "external output directory",secondary_csv='data/operator-setup-plus-solver-statistics.csv')
write('BUILD-CHECKS.json',json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))

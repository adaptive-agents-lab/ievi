"""Validate and tabulate incumbent-restoration common-solver control measurements."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from control_accuracy_common import stable_hit_index,first_hit_index

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);a=p.parse_args();a.artifacts.mkdir(parents=True,exist_ok=False)
 new=pd.read_csv(a.results/'control_accuracy_summary.csv');old=pd.read_csv(a.archive/'results/benchmarks/control_accuracy/raw/control_accuracy_summary.csv')
 traj=pd.read_csv(a.results/'control_accuracy_ievi_trajectories.csv');att=pd.read_csv(a.results/'control_accuracy_ievi_attempts.csv')
 keys=['lambda','algorithm','delta_solve'];merged=old.merge(new,on=keys,suffixes=('_old','_new'),validate='one_to_one')
 assert len(merged)==95
 metrics=['stable_N_P','stable_N_Phat','first_hit_output','stable_hit_output','final_N_P','final_N_Phat','final_attempts','final_accepted','final_rejected','final_inner_failures','final_surrogate_vi_sweeps']
 comp=merged[keys].copy()
 for field in metrics:
  comp[field+'_old']=merged[field+'_old'];comp[field+'_new']=merged[field+'_new'];comp[field+'_difference']=merged[field+'_new']-merged[field+'_old']
 comp['status_old']=merged.status_old;comp['status_new']=merged.status_new
 comp.to_csv(a.results/'all-95-row-comparison.csv',index=False)
 for field in old.columns:
  if field in keys or field=='wall_seconds_full_run':continue
  subset=merged[merged.algorithm!='IEVI_Waterfall_common_VI']
  assert ((subset[field+'_old']==subset[field+'_new'])|(subset[field+'_old'].isna()&subset[field+'_new'].isna())).all(),field
 checks=[]
 for _,r in new[new.algorithm=='IEVI_Waterfall_common_VI'].iterrows():
  t=traj[(traj['lambda']==r['lambda'])&(traj.delta_solve==r.delta_solve)];u=att[(att['lambda']==r['lambda'])&(att.delta_solve==r.delta_solve)]
  stable=stable_hit_index(t.policy_value_error);first=first_hit_index(t.policy_value_error)
  assert r.status=='completed_budget' and len(t)==999 and len(u)==999
  assert t.iloc[stable].N_P==r.stable_N_P and t.iloc[stable].N_Phat==r.stable_N_Phat
  assert first+1==r.first_hit_output and stable+1==r.stable_hit_output
  assert r.final_N_P==1+u.accepted.sum()==1000
  assert r.final_N_Phat==u.inner_sweeps.sum()==r.final_surrogate_vi_sweeps
  assert np.all(u.inner_value_error_bound<=r.delta_solve)
  assert np.all(np.diff(u.incumbent_residual)<=0)
  decreased=u.h_after<u.h_before
  assert np.array_equal(decreased,u.restored_incumbent)
  assert np.all(u.loc[decreased,'current_residual_after']==u.loc[decreased,'incumbent_residual'])
  assert u.restored_incumbent.sum()==r.restoration_events
  checks.append({'lambda':r['lambda'],'delta_solve':r.delta_solve,'checked_outputs':len(t),'restorations':int(decreased.sum()),'stable_true_residual':float(t.iloc[stable].true_bellman_residual),'stable_value_error_bound_from_true_residual':float(t.iloc[stable].true_bellman_residual/.01),'final_true_residual':float(t.iloc[-1].true_bellman_residual),'max_inner_value_error_bound':float(u.inner_value_error_bound.max())})
 ie=new[(new.algorithm=='IEVI_Waterfall_common_VI')&(new.delta_solve==.01)].sort_values('lambda');tight=new[(new.algorithm=='IEVI_Waterfall_common_VI')&(new.delta_solve==1e-8)].sort_values('lambda');osvi=new[(new.algorithm=='OSVI_common_VI')&(new.delta_solve==.01)].sort_values('lambda')
 ratio=tight.stable_N_Phat.to_numpy()/ie.stable_N_Phat.to_numpy();assert np.array_equal(tight.stable_N_P,ie.stable_N_P)
 iecomp=comp[comp.algorithm=='IEVI_Waterfall_common_VI']
 summary={'recomputed_runs':38,'retained_rows':57,'total_rows':95,'all_recomputed_runs_complete_budget':True,'all_inner_value_error_bounds_certified':True,'all_recomputed_IEVI_stable_hits':38,'OSVI_stable_hits_per_accuracy':int(osvi.stable_N_P.notna().sum()),'VI_stable_N_P':49,'true_counts_equal_across_accuracies':True,'median_tight_over_loose_surrogate_work':float(np.median(ratio)),'old_median_tight_over_loose_surrogate_work':2.613275,'changed_IEVI_stable_true_count_rows':int((iecomp.stable_N_P_difference!=0).sum()),'improved_IEVI_stable_true_count_rows':int((iecomp.stable_N_P_difference<0).sum()),'worsened_IEVI_stable_true_count_rows':int((iecomp.stable_N_P_difference>0).sum()),'unchanged_IEVI_stable_true_count_rows':int((iecomp.stable_N_P_difference==0).sum()),'total_restorations':int(att.restored_incumbent.sum()),'restoration_types':att[att.restored_incumbent].restoration_reason.value_counts().to_dict(),'total_true_applications':int(new[new.algorithm=='IEVI_Waterfall_common_VI'].final_N_P.sum()),'total_surrogate_applications':int(new[new.algorithm=='IEVI_Waterfall_common_VI'].final_N_Phat.sum()),'loose_break_even_positive_lambdas':ie[ie.stable_N_P<49]['lambda'].tolist(),'loose_beats_VI_kappa_0p001_lambdas':ie[ie.stable_N_P+.001*ie.stable_N_Phat<49]['lambda'].tolist(),'loose_beats_VI_kappa_0p01_lambdas':ie[ie.stable_N_P+.01*ie.stable_N_Phat<49]['lambda'].tolist(),'validation_by_run':checks,'new_summary_rows':json.loads(new.to_json(orient='records')),'comparison_rows':json.loads(comp.to_json(orient='records'))}
 (a.results/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 # Generate table bodies using the paper labels.
 def num(x):return '---' if pd.isna(x) else rf'\({int(x):,}\)'.replace(',',r'{,}')
 tables={'counts':[],'costs':[],'tolerance-ievi':[]}
 for i,(_,r) in enumerate(ie.iterrows()):
  o=osvi.iloc[i];t=tight.iloc[i];lam=f'{r["lambda"]:g}'
  tables['counts'].append(' & '.join([rf'\({lam}\)',num(o.stable_N_P),num(o.stable_N_Phat),num(r.stable_N_P),num(r.stable_N_Phat)])+r' \\')
  k='None' if pd.isna(r.kappa_star_vs_vi) else rf'\({r.kappa_star_vs_vi:.6f}\)'
  tables['costs'].append(' & '.join([rf'\({lam}\)',k,rf'\({(r.stable_N_P+.001*r.stable_N_Phat)/49:.4f}\)',rf'\({(r.stable_N_P+.01*r.stable_N_Phat)/49:.4f}\)'])+r' \\')
  tables['tolerance-ievi'].append(' & '.join([rf'\({lam}\)',num(r.stable_N_P),num(t.stable_N_Phat),num(r.stable_N_Phat),rf'\({ratio[i]:.3f}\)'])+r' \\')
 for name,rows in tables.items():(a.artifacts/f'table-{name}-rows.tex').write_text('\n'.join(rows)+'\n')
 for name in ['control_accuracy_summary.csv','all-95-row-comparison.csv','summary.json','effective-config.json','provenance.json']:shutil.copy2(a.results/name,a.artifacts/name)
 (a.artifacts/'ARTIFACT-MANIFEST.json').write_text(json.dumps({'producing_script':str(Path(__file__).resolve()),'producing_script_sha256':sha(Path(__file__)),'input_summary_sha256':sha(a.results/'control_accuracy_summary.csv'),'tables':{'counts':'table-counts-rows.tex','costs':'table-costs-rows.tex','tolerance':'table-tolerance-ievi-rows.tex'},'artifact_sha256':{x.name:sha(x) for x in a.artifacts.iterdir() if x.is_file()}},indent=2)+'\n')
 baseline=merged[merged.algorithm!='IEVI_Waterfall_common_VI']
 timing_equal=(baseline.wall_seconds_full_run_old==baseline.wall_seconds_full_run_new)
 timing_scope={'rows':len(baseline),'equal_full_run_observations':int(timing_equal.sum()),'timing_equality_required':False,'timing_reported_in_Table_N11':False,'recorded_timing_values_preserved':True,'note':'Table N.11 compares operator-query costs. Full-budget elapsed observations from the two recorded sources are not treated as a timing comparison.'}
 print(json.dumps({**{k:v for k,v in summary.items() if k not in ['validation_by_run','new_summary_rows','comparison_rows']},'baseline_timing_scope':timing_scope},indent=2))
if __name__=='__main__':main()

"""Compute incumbent-restoration IE-VI measurements alongside supplied VI/OS-VI reference measurements."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import run_control_accuracy_common_solver as m
from control_accuracy_common import *

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 a.output_dir.mkdir(parents=True,exist_ok=False)
 here=Path(__file__).parent;assets=here/'assets/target_arrays.npz'
 P,R=build_modified_cliffwalk()
 with np.load(assets) as z:target=z['v_star_control']
 oldraw=a.archive/'results/benchmarks/control_accuracy/raw';summ=[];traj=[];attempts=[]
 oldsum=read(oldraw/'control_accuracy_summary.csv')
 for lam in LAMBDAS:
  Ph=support_uniform_surrogate(P,lam);cache=PolicyErrorCache(P,R,GAMMA,target)
  for delta in m.DELTA_LEVELS:
   outs,atts,c,status,elapsed=m.run_waterfall(P,Ph,R,target,cache,delta)
   for row in outs+atts:row['lambda']=lam
   summary=m.finalize_summary('control_accuracy',lam,'IEVI_Waterfall_common_VI',delta,outs,c,status,elapsed,49)
   summary.update(final_true_bellman_residual=outs[-1]['true_bellman_residual'],final_incumbent_residual=atts[-1]['incumbent_residual'],restoration_events=sum(x['restored_incumbent'] for x in atts),max_inner_value_error_bound=max(x['inner_value_error_bound'] for x in atts),data_source='incumbent_restoration')
   summ.append(summary);traj+=outs;attempts+=atts
   m.write_csv(a.output_dir/f'lambda-{lam:g}-delta-{delta:g}-summary.csv',[summary])
   print(json.dumps({k:summary[k] for k in ['lambda','delta_solve','stable_N_P','stable_N_Phat','restoration_events','status']}),flush=True)
 retained=[dict(row,data_source='reference_data') for row in oldsum if row['algorithm']!='IEVI_Waterfall_common_VI']
 m.write_csv(a.output_dir/'control_accuracy_ievi_summary.csv',summ)
 m.write_csv(a.output_dir/'control_accuracy_summary.csv',retained+summ)
 m.write_csv(a.output_dir/'control_accuracy_ievi_trajectories.csv',traj)
 m.write_csv(a.output_dir/'control_accuracy_ievi_attempts.csv',attempts)
 config={'lambdas':LAMBDAS,'gamma':GAMMA,'delta_solve_levels':m.DELTA_LEVELS,'true_budget':TRUE_BUDGET,'policy_threshold_strict':POLICY_THRESHOLD,'h0':1e6,'h_floor':1.,'decay':.1,'stall_length':3,'outer_attempt_cap':5000,'inner_sweep_cap':5000,'divergence_mult':DIVERGENCE_MULT,'precision':'float64','inner_stop':'||T_hat U-U||_inf <= (1-beta)*delta_solve','policy_ties':'first-index argmax','seed':'none; deterministic','policy_semantics':'associated inner-solve policy saved/restored with value; initial policy true-greedy at zero','restoration':'strict improvement only; restore V,PV,residual,policy only on actual h reduction; no operator cost; retain old guard floor-failure and inner-cap semantics'}
 write_json(a.output_dir/'effective-config.json',config)
 sources=[here/'control_accuracy_common.py',here/'run_control_accuracy_common_solver.py',here/'run.py',assets,oldraw/'control_accuracy_summary.csv',oldraw/'control_accuracy_trajectories.csv',oldraw/'control_accuracy_ievi_attempts.csv',a.archive/'benchmarks/src/ievi_experiments/control_accuracy/control_accuracy_common.py',a.archive/'benchmarks/src/ievi_experiments/control_accuracy/run_control_accuracy_common_solver.py',a.output_dir/'effective-config.json']
 write_json(a.output_dir/'provenance.json',{'argv':sys.argv,'executable':sys.executable,'environment':environment_record(),'source_sha256':{str(x):digest(x) for x in sources},'archived_raw':str(oldraw),'retained_algorithms':['VI_current','OSVI_common_VI'],'recomputed_runs':38,'reference_timing_retained':True,'note':'IE-VI elapsed times are diagnostic full-run times only, not substitutes for independent timing studies.'})
if __name__=='__main__':main()

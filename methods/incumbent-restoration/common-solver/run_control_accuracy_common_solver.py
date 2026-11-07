#!/usr/bin/env python3
"""Compare solve accuracy across the full parameter panel using a common solver."""
from __future__ import annotations
import argparse,csv,json,math,sys,time
from pathlib import Path
from typing import Any
import numpy as np
HERE=Path(__file__).resolve();sys.path.insert(0,str(HERE.parent))
from control_accuracy_common import (
 ANCHOR_LAMBDAS,DIVERGENCE_MULT,GAMMA,LAMBDAS,POLICY_THRESHOLD,TRUE_BUDGET,
 Counters,PolicyErrorCache,build_modified_cliffwalk,certified_surrogate_vi_control,
 first_hit_index,stable_hit_index,support_uniform_surrogate,true_bellman_residual,
 environment_record,write_json,
)
DELTA_LEVELS=[1e-8,1e-2]
KAPPAS=[0.0,1e-4,1e-3,1e-2,1e-1,1.0]

def parse():
 p=argparse.ArgumentParser();p.add_argument('--target-arrays',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);return p.parse_args()
def write_csv(path,rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def finalize_summary(track,lam,alg,delta,outputs,c,status,elapsed,vi_np):
 errs=[x['policy_value_error'] for x in outputs];fh=first_hit_index(errs);sh=stable_hit_index(errs)
 if status!='completed_budget': sh=None
 sr=None if sh is None else outputs[sh]
 row={'track':track,'lambda':lam,'algorithm':alg,'delta_solve':delta,'status':status,'completed_outputs':len(outputs),'first_hit_output':None if fh is None else fh+1,'stable_hit_output':None if sh is None else sh+1,'stable_N_P':None if sr is None else sr['N_P'],'stable_N_Phat':None if sr is None else sr['N_Phat'],'stable_direct_solves':None if sr is None else sr.get('direct_solves',0),'stable_attempts':None if sr is None else sr.get('attempt',sr.get('outer',None)),'final_N_P':c.n_p,'final_N_Phat':c.n_phat,'final_direct_solves':c.direct_solves,'final_attempts':c.attempts,'final_accepted':c.accepted,'final_rejected':c.rejected,'final_inner_failures':c.inner_failures,'final_surrogate_vi_sweeps':c.surrogate_vi_sweeps,'wall_seconds_full_run':elapsed,'kappa_star_vs_vi':None}
 if sr is not None and sr['N_P']<vi_np and sr['N_Phat']>0:row['kappa_star_vs_vi']=(vi_np-sr['N_P'])/sr['N_Phat']
 for kap in KAPPAS:
  row[f'C_kappa_{kap:g}']=None if sr is None else sr['N_P']+kap*sr['N_Phat']
 return row

def run_vi(P,R,target,cache):
 c=Counters();v=np.zeros(P.shape[0]);outs=[];t=time.perf_counter()
 for q in range(1,TRUE_BUDGET+1):
  pv=np.einsum('sak,k->sa',P,v);c.n_p+=1;c.attempts+=1;c.accepted+=1
  qq=R+GAMMA*pv;pi=np.argmax(qq,axis=1).astype(np.int32);v=np.max(qq,axis=1)
  outs.append({'track':'control_accuracy','lambda':None,'algorithm':'VI_current','delta_solve':None,'outer':q,'attempt':q,'N_P':c.n_p,'N_Phat':c.n_phat,'direct_solves':0,'policy_value_error':cache.error(pi)})
 return outs,c,'completed_budget',time.perf_counter()-t

def run_osvi(P,Ph,R,target,cache,delta):
 c=Counters();v=np.zeros(P.shape[0]);outs=[];vmax=DIVERGENCE_MULT*max(np.max(np.abs(R))/(1-GAMMA),1.0);status='completed_budget';t=time.perf_counter()
 while c.n_p<TRUE_BUDGET:
  c.attempts+=1
  pv=np.einsum('sak,k->sa',P,v);phv=np.einsum('sak,k->sa',Ph,v);c.n_p+=1;c.n_phat+=1
  corrected=R+GAMMA*(pv-phv)
  sol=certified_surrogate_vi_control(Ph,corrected,GAMMA,v,delta,initial_phat_value=phv,max_sweeps=5000)
  c.n_phat+=sol.new_phat_apps;c.surrogate_vi_sweeps+=sol.sweeps
  if not sol.converged:
   c.inner_failures+=1;status='inner_solver_cap';break
  v=sol.value
  if not np.all(np.isfinite(v)) or np.max(np.abs(v))>vmax:
   status='divergence_guard';break
  c.accepted+=1
  outs.append({'track':'control_accuracy','lambda':None,'algorithm':'OSVI_common_VI','delta_solve':delta,'outer':len(outs)+1,'attempt':c.attempts,'N_P':c.n_p,'N_Phat':c.n_phat,'direct_solves':0,'policy_value_error':cache.error(sol.policy),'inner_sweeps':sol.sweeps,'inner_residual':sol.residual,'inner_value_error_bound':sol.value_error_bound})
 return outs,c,status,time.perf_counter()-t

def run_waterfall(P,Ph,R,target,cache,delta):
 """Restore the strict best-residual tuple on each actual step reduction.

 Candidate certificates describe the attempted solve; output residual/policy
 describe the carried state after restoration. Cached restoration costs no
 operator applications. The incumbent's policy is the associated inner-solve
 policy (initially the true greedy policy at zero). Floor and failure rules
 are unchanged from the archived experiment.
 """
 c=Counters();v=np.zeros(P.shape[0]);pv=np.einsum('sak,k->sa',P,v);c.n_p=1
 best,_=true_bellman_residual(P,R,GAMMA,v,pv);current=best
 last_policy=np.argmax(R+GAMMA*pv,axis=1).astype(np.int32)
 best_v=v.copy();best_pv=pv.copy();best_policy=last_policy.copy();best_attempt=0
 h=1e6;hmin=1.;stall=0;outs=[];attempts=[];restorations=0
 vmax=DIVERGENCE_MULT*max(np.max(np.abs(R))/(1-GAMMA),1.0);status='completed_budget';t=time.perf_counter()
 while c.n_p<TRUE_BUDGET and c.attempts<5000:
  c.attempts+=1;phv=np.einsum('sak,k->sa',Ph,v);c.n_phat+=1;beta=h*GAMMA/(1+h);shifted=v[:,None]/(1+h)+h*(R+GAMMA*(pv-phv))/(1+h)
  sol=certified_surrogate_vi_control(Ph,shifted,beta,v,delta,initial_phat_value=phv,max_sweeps=5000);c.n_phat+=sol.new_phat_apps;c.surrogate_vi_sweeps+=sol.sweeps
  attempt={'track':'control_accuracy','lambda':None,'algorithm':'IEVI_Waterfall_common_VI','delta_solve':delta,'attempt':c.attempts,'N_P_before_accept':c.n_p,'N_Phat_after_solve':c.n_phat,'h_before':h,'inner_sweeps':sol.sweeps,'inner_residual':sol.residual,'inner_value_error_bound':sol.value_error_bound,'accepted':False,'rejection_reason':None,'restored_incumbent':False,'restoration_reason':None,'current_residual_before':current}
  if not sol.converged:
   c.inner_failures+=1;c.rejected+=1;attempt['rejection_reason']='inner_solver_cap';attempts.append(attempt);status='inner_solver_cap';break
  candidate=sol.value
  if not np.all(np.isfinite(candidate)) or np.max(np.abs(candidate))>vmax:
   c.rejected+=1;attempt['rejection_reason']='divergence_guard';newh=max(hmin,.1*h);stall=0
   if newh<h:
    v=best_v.copy();pv=best_pv.copy();current=best;last_policy=best_policy.copy();restorations+=1
    attempt['restored_incumbent']=True;attempt['restoration_reason']='guard'
   h=newh;attempt.update(h_after=h,current_residual_after=current,incumbent_attempt=best_attempt,incumbent_residual=best,restorations=restorations);attempts.append(attempt)
   if h<=hmin:status='failed_at_h_floor';break
   continue
  pvnew=np.einsum('sak,k->sa',P,candidate);c.n_p+=1;newres,_=true_bellman_residual(P,R,GAMMA,candidate,pvnew)
  v=candidate;pv=pvnew;current=newres;last_policy=sol.policy;c.accepted+=1;attempt['accepted']=True;attempt['N_P_after_accept']=c.n_p;attempt['true_residual']=newres
  if newres<best:
   best=newres;best_v=v.copy();best_pv=pv.copy();best_policy=last_policy.copy();best_attempt=c.attempts;stall=0
  else:
   stall+=1
   if stall>=3:
    newh=max(hmin,.1*h);stall=0
    if newh<h:
     v=best_v.copy();pv=best_pv.copy();current=best;last_policy=best_policy.copy();restorations+=1
     attempt['restored_incumbent']=True;attempt['restoration_reason']='stall'
    h=newh
  attempt.update(h_after=h,current_residual_after=current,incumbent_attempt=best_attempt,incumbent_residual=best,restorations=restorations);attempts.append(attempt)
  outs.append({'track':'control_accuracy','lambda':None,'algorithm':'IEVI_Waterfall_common_VI','delta_solve':delta,'outer':len(outs)+1,'attempt':c.attempts,'N_P':c.n_p,'N_Phat':c.n_phat,'direct_solves':0,'policy_value_error':cache.error(last_policy),'true_bellman_residual':current,'candidate_true_bellman_residual':newres,'incumbent_true_bellman_residual':best,'incumbent_attempt':best_attempt,'restored_incumbent':attempt['restored_incumbent'],'restorations':restorations,'h':h,'inner_sweeps':sol.sweeps,'inner_residual':sol.residual,'inner_value_error_bound':sol.value_error_bound,'accepted_attempts':c.accepted,'rejected_attempts':c.rejected})
 if c.n_p<TRUE_BUDGET and status=='completed_budget':status='attempt_cap'
 return outs,attempts,c,status,time.perf_counter()-t

def main():
 a=parse();a.output_dir.mkdir(parents=True,exist_ok=True);P,R=build_modified_cliffwalk()
 with np.load(a.target_arrays) as z:target=np.asarray(z['v_star_control'],float)
 allsum=[];alltraj=[];allattempt=[]
 # VI is deterministic and independent of lambda, but execute it afresh in each paired condition.
 for lam in LAMBDAS:
  Ph=support_uniform_surrogate(P,lam);cache=PolicyErrorCache(P,R,GAMMA,target)
  viouts,vic,vis,vsec=run_vi(P,R,target,cache)
  for o in viouts:o['lambda']=lam
  visum=finalize_summary('control_accuracy',lam,'VI_current',None,viouts,vic,vis,vsec,49);allsum.append(visum);alltraj+=viouts;vi_np=int(visum['stable_N_P'])
  for delta in DELTA_LEVELS:
   oo,oc,ost,osec=run_osvi(P,Ph,R,target,cache,delta)
   for o in oo:o['lambda']=lam
   allsum.append(finalize_summary('control_accuracy',lam,'OSVI_common_VI',delta,oo,oc,ost,osec,vi_np));alltraj+=oo
   wo,wa,wc,wst,wsec=run_waterfall(P,Ph,R,target,cache,delta)
   for o in wo:o['lambda']=lam
   for x in wa:x['lambda']=lam
   allsum.append(finalize_summary('control_accuracy',lam,'IEVI_Waterfall_common_VI',delta,wo,wc,wst,wsec,vi_np));alltraj+=wo;allattempt+=wa
  print('lambda',lam,'done',flush=True)
 write_csv(a.output_dir/'control_accuracy_summary.csv',allsum);write_csv(a.output_dir/'control_accuracy_trajectories.csv',alltraj);write_csv(a.output_dir/'control_accuracy_ievi_attempts.csv',allattempt)
 write_json(a.output_dir/'control_accuracy_metadata.json',{'description':'Matched full-panel comparison. OS-VI and practical IE-VI share the same surrogate VI solver, warm start, sweep cap, first-index tie breaking, and certified absolute L-infinity value-accuracy targets.','delta_solve_levels':DELTA_LEVELS,'inner_stop':'||T_hat U-U||_inf <= (1-beta) delta_solve','true_budget':TRUE_BUDGET,'threshold_strict':POLICY_THRESHOLD,'symmetric_caching':'P_hat@initial_value is shared between defect construction and the first surrogate Bellman sweep','environment':environment_record(),'rows_summary':len(allsum),'rows_trajectory':len(alltraj),'rows_attempts':len(allattempt)})
if __name__=='__main__':main()

"""Independent diagnostic replay: inspect each restore without editing algorithm.

Diagnostic recomputations are offline audits, excluded from algorithm counters.
Tracing observes function locals immediately after restoration; it never mutates
those locals. All candidate residuals are independently recomputed.
"""
import csv,inspect,json,sys
from pathlib import Path
import numpy as np
import run_control_accuracy_common_solver as m
from control_accuracy_common import *
from unittest.mock import patch

def main():
 here=Path(__file__).parent;results=here.parent/'results/common-solver';outdir=results/'restore-state-audit';outdir.mkdir(exist_ok=False)
 with (results/'control_accuracy_ievi_summary.csv').open() as f:reference=list(csv.DictReader(f))
 with np.load(here/'assets/target_arrays.npz') as z:target=z['v_star_control']
 P,R=build_modified_cliffwalk();events=[];checks=[]
 original_residual=m.true_bellman_residual;original_solver=m.certified_surrogate_vi_control
 lines,start=inspect.getsourcelines(m.run_waterfall)
 capture_lines={start+i for i,line in enumerate(lines) if 'attempt.update(h_after=' in line}
 for lam in LAMBDAS:
  Ph=support_uniform_surrogate(P,lam)
  for delta in m.DELTA_LEVELS:
   history=[];best=None;candidate_policy=None;run_events=[]
   def solve(*args,**kwargs):
    nonlocal candidate_policy
    result=original_solver(*args,**kwargs);candidate_policy=result.policy.copy();return result
   def residual(Parg,Rarg,gamma,value,pv=None):
    nonlocal best
    independent_pv=np.einsum('sak,k->sa',Parg,value)
    independent_residual=float(np.max(np.abs(np.max(Rarg+gamma*independent_pv,axis=1)-value)))
    np.testing.assert_array_equal(pv,independent_pv)
    reported,cached=original_residual(Parg,Rarg,gamma,value,pv)
    assert reported==independent_residual
    pi=np.argmax(Rarg+gamma*independent_pv,axis=1).astype(np.int32) if not history else candidate_policy.copy()
    item={'history_index':len(history),'value':value.copy(),'pv':independent_pv.copy(),'residual':independent_residual,'policy':pi}
    history.append(item)
    if best is None or independent_residual<best['residual']:best=item
    return reported,cached
   def trace(frame,event,arg):
    if frame.f_code is not m.run_waterfall.__code__:return None
    if event=='line' and frame.f_lineno in capture_lines:
     state=frame.f_locals;attempt=state['attempt']
     if attempt['restored_incumbent']:
      # Independent strict argmin over the history, not the solver's best tuple.
      winner=min(history,key=lambda item:item['residual'])
      assert winner['history_index']==best['history_index']
      for actual,expected in [('v','value'),('pv','pv'),('last_policy','policy')]:np.testing.assert_array_equal(state[actual],winner[expected])
      assert state['current']==winner['residual']
      recomputed_pv=np.einsum('sak,k->sa',P,state['v'])
      recomputed_residual=float(np.max(np.abs(np.max(R+GAMMA*recomputed_pv,axis=1)-state['v'])))
      np.testing.assert_array_equal(state['pv'],recomputed_pv);assert state['current']==recomputed_residual
      assert state['c'].n_p==len(history)  # initialization + accepted candidates only
      run_events.append({'lambda':lam,'delta_solve':delta,'attempt':state['c'].attempts,'reason':attempt['restoration_reason'],'reference_values_checked':len(history),'strict_argmin_history_index':winner['history_index'],'true_residual':state['current'],'N_P':state['c'].n_p,'N_Phat':state['c'].n_phat,'restored_value':state['v'].tolist(),'restored_pv':state['pv'].tolist(),'restored_policy':state['last_policy'].tolist(),'all_checks_exact':True})
    return trace
   with patch.object(m,'true_bellman_residual',residual),patch.object(m,'certified_surrogate_vi_control',solve):
    sys.settrace(trace)
    try:outputs,attempts,c,status,elapsed=m.run_waterfall(P,Ph,R,target,PolicyErrorCache(P,R,GAMMA,target),delta)
    finally:sys.settrace(None)
   summary=m.finalize_summary('control_accuracy',lam,'IEVI_Waterfall_common_VI',delta,outputs,c,status,elapsed,49)
   ref=next(r for r in reference if float(r['lambda'])==lam and float(r['delta_solve'])==delta)
   for key,val in summary.items():
    if key=='wall_seconds_full_run':continue
    old=ref[key]
    assert (old=='' and val is None) or old==str(val) or (isinstance(val,(int,float)) and float(old)==val),(key,old,val)
   assert len(run_events)==int(ref['restoration_events'])
   checks.append({'lambda':lam,'delta_solve':delta,'independent_candidate_residuals_checked':len(history),'restores_checked':len(run_events),'all_non_timing_summary_fields_equal':True,'algorithm_N_P':c.n_p,'algorithm_N_Phat':c.n_phat})
   events+=run_events
   print(f'lambda={lam:g} delta={delta:g}: {len(history)} reference values, {len(run_events)} restores exact, summary parity',flush=True)
 write_json(outdir/'restoration-state-evidence.json',{'diagnostic_only':True,'events':events})
 write_json(outdir/'checks.json',{'passed':True,'runs':checks,'total_restores':len(events),'all_checks_exact':True,'diagnostic_P_applications':sum(x['independent_candidate_residuals_checked']+x['restores_checked'] for x in checks),'diagnostic_applications_excluded_from_algorithm_cost':True,'replay_algorithm_true_apps':sum(x['algorithm_N_P'] for x in checks),'replay_algorithm_surrogate_apps':sum(x['algorithm_N_Phat'] for x in checks),'instrumentation':'read-only sys.settrace at post-restoration local state; independent candidate residuals and strict history argmin; solver wrapper captures each returned candidate policy'})
if __name__=='__main__':main()

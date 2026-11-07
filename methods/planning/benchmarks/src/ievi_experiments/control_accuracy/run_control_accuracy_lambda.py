#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve();sys.path.insert(0,str(HERE.parent))
from control_accuracy_common import GAMMA,PolicyErrorCache,build_modified_cliffwalk,support_uniform_surrogate,write_json
from run_control_accuracy_common_solver import DELTA_LEVELS,finalize_summary,run_vi,run_osvi,run_waterfall

def write_csv(path,rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
p=argparse.ArgumentParser();p.add_argument('--target-arrays',type=Path,required=True);p.add_argument('--lambda-value',type=float,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
P,R=build_modified_cliffwalk();lam=float(a.lambda_value);Ph=support_uniform_surrogate(P,lam)
with np.load(a.target_arrays) as z:target=np.asarray(z['v_star_control'],float)
cache=PolicyErrorCache(P,R,GAMMA,target);sums=[];trajs=[];attempts=[]
vo,vc,vst,vsec=run_vi(P,R,target,cache)
for x in vo:x['lambda']=lam
vs=finalize_summary('control_accuracy',lam,'VI_current',None,vo,vc,vst,vsec,49);sums.append(vs);trajs+=vo;vi_np=int(vs['stable_N_P'])
for d in DELTA_LEVELS:
 oo,oc,ost,osec=run_osvi(P,Ph,R,target,cache,d)
 for x in oo:x['lambda']=lam
 sums.append(finalize_summary('control_accuracy',lam,'OSVI_common_VI',d,oo,oc,ost,osec,vi_np));trajs+=oo
 wo,wa,wc,wst,wsec=run_waterfall(P,Ph,R,target,cache,d)
 for x in wo:x['lambda']=lam
 for x in wa:x['lambda']=lam
 sums.append(finalize_summary('control_accuracy',lam,'IEVI_Waterfall_common_VI',d,wo,wc,wst,wsec,vi_np));trajs+=wo;attempts+=wa
write_csv(a.output_dir/'summary.csv',sums);write_csv(a.output_dir/'trajectories.csv',trajs);write_csv(a.output_dir/'attempts.csv',attempts);write_json(a.output_dir/'summary.json',sums)
print('lambda',lam,[(r['algorithm'],r['delta_solve'],r['stable_N_P'],r['stable_N_Phat'],r['status']) for r in sums],flush=True)

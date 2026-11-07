#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
HERE=Path(__file__).resolve();sys.path.insert(0,str(HERE.parent))
from control_accuracy_common import LAMBDAS,environment_record,write_json
p=argparse.ArgumentParser();p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
sums=[];tr=[];at=[]
for lam in LAMBDAS:
 d=a.input_root/f'lambda_{lam:g}';sums+=json.loads((d/'summary.json').read_text())
 for fn,dst in [('trajectories.csv',tr),('attempts.csv',at)]:
  with (d/fn).open() as f:dst+=list(csv.DictReader(f))
def wc(path,rows):
 fields=[]
 for r in rows:
  for k in r:
   if k not in fields:fields.append(k)
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
wc(a.output_dir/'control_accuracy_summary.csv',sums);wc(a.output_dir/'control_accuracy_trajectories.csv',tr);wc(a.output_dir/'control_accuracy_ievi_attempts.csv',at)
write_json(a.output_dir/'control_accuracy_metadata.json',{'description':'Matched full-panel comparison. OS-VI and practical IE-VI share the same surrogate VI solver, warm start, sweep cap, tie breaking, certified solve accuracy, and symmetric initial P_hat caching.','delta_solve_levels':[1e-8,1e-2],'inner_stop':'||T_hat U-U||_inf <= (1-beta) delta_solve','environment':environment_record(),'rows_summary':len(sums),'rows_trajectory':len(tr),'rows_attempts':len(at)})
print('aggregated',len(sums),len(tr),len(at))

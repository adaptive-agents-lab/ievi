#!/usr/bin/env python3
"""Independent algebraic audit of the policy-evaluation horizon study.

This script does not call the experiment's iterative algorithm implementations.
Instead, it constructs each method's exact linear error-propagation matrix and
recomputes the stable-hit query count from e_k = G^k e_0.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); return p.parse_args()


def relative(e,target):
    return float(np.max(np.abs(e))/max(float(np.max(np.abs(target))),1e-300))


def stable_hit_from_matrix(G,target,tol,tail,cap_outer):
    e=-target.copy(); first=None; streak=0; err_at_hit=math.inf; err=relative(e,target)
    for k in range(1,cap_outer+1):
        e=G@e; err=relative(e,target)
        old=first
        if err<=tol:
            if first is None: first=k
            streak+=1
        else:
            first=None; streak=0
        if old is None and first is not None: err_at_hit=err
        if streak>=tail: return True,first,k,err_at_hit,err
    return False,cap_outer,cap_outer,math.inf,err


def os_matrix(P,Ph,gamma):
    return np.linalg.solve(np.eye(P.shape[0])-gamma*Ph,gamma*(P-Ph))


def ie_matrix(P,Ph,gamma,h,m):
    S=P.shape[0]
    Ainv=np.linalg.inv(np.eye(S)-(h*gamma/(1+h))*Ph)
    C=Ainv/(1+h)
    D=Ainv@(h*gamma*(P-Ph)/(1+h))
    M=np.linalg.matrix_power(D,m)
    Dp=np.eye(S)
    for _ in range(m):
        M=M+Dp@C; Dp=Dp@D
    return M


def hsh(a): return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

def main():
    a=parse_args(); root=a.root.resolve()
    z=np.load(root/'data/raw/experiment_inputs.npz',allow_pickle=False)
    P=z['P_policy']; r=z['r_policy']; targets=z['targets']; Hs=z['H_grid']; models=z['surrogate_models']; schedules=[str(x) for x in z['schedules']]
    df=pd.read_csv(root/'data/raw/records.csv')
    mismatches=[]; envelope_failures=[]; checked=0
    for row in df.itertuples(index=False):
        rep=int(row.replication); hi=int(np.where(Hs==int(row.H))[0][0]); si=schedules.index(row.schedule)
        Ph=models[rep,si,hi]; target=targets[hi]; gamma=float(row.gamma); m=int(row.m)
        if row.algorithm=='VI': G=gamma*P
        elif row.algorithm=='OS-VI': G=os_matrix(P,Ph,gamma)
        else: G=ie_matrix(P,Ph,gamma,float(row.h),m)
        actual_inf=float(np.linalg.norm(G,ord=np.inf)); actual_sr=float(np.max(np.abs(np.linalg.eigvals(G))))
        if abs(actual_inf-float(row.actual_inf_norm))>1e-9 or abs(actual_sr-float(row.actual_spectral_radius))>1e-9:
            mismatches.append({'kind':'matrix_stat','rep':rep,'H':int(row.H),'schedule':row.schedule,'algorithm':row.algorithm,'stored_inf':float(row.actual_inf_norm),'audit_inf':actual_inf,'stored_sr':float(row.actual_spectral_radius),'audit_sr':actual_sr})
        if row.algorithm.startswith('IE-VI') and actual_inf>float(row.mu_envelope)+2e-10:
            envelope_failures.append({'rep':rep,'H':int(row.H),'schedule':row.schedule,'algorithm':row.algorithm,'actual_inf':actual_inf,'mu':float(row.mu_envelope)})
        if row.algorithm=='OS-VI' and actual_inf>float(row.R_E)+2e-10:
            envelope_failures.append({'rep':rep,'H':int(row.H),'schedule':row.schedule,'algorithm':row.algorithm,'actual_inf':actual_inf,'R_E':float(row.R_E)})
        cap_outer=250000 if m==1 else 250000//m
        hit,first,end,err_hit,err_end=stable_hit_from_matrix(G,target,float(row.target_relative_linf),int(row.stable_tail_outer_updates),cap_outer)
        q=first*m; qend=end*m
        if hit!=bool(row.hit) or q!=int(row.stable_hit_true_applications) or qend!=int(row.verification_true_applications):
            mismatches.append({'kind':'stable_hit','rep':rep,'H':int(row.H),'schedule':row.schedule,'algorithm':row.algorithm,'stored_hit':bool(row.hit),'audit_hit':hit,'stored_q':int(row.stable_hit_true_applications),'audit_q':q,'stored_qend':int(row.verification_true_applications),'audit_qend':qend})
        if hit and (abs(err_hit-float(row.error_at_hit))>1e-8 or abs(err_end-float(row.error_after_stable_tail))>1e-8):
            mismatches.append({'kind':'error','rep':rep,'H':int(row.H),'schedule':row.schedule,'algorithm':row.algorithm,'stored_hit_error':float(row.error_at_hit),'audit_hit_error':err_hit,'stored_end_error':float(row.error_after_stable_tail),'audit_end_error':err_end})
        if hsh(Ph)!=row.surrogate_model_sha256:
            mismatches.append({'kind':'model_hash','rep':rep,'H':int(row.H),'schedule':row.schedule,'algorithm':row.algorithm})
        checked+=1
    timings=pd.read_csv(root/'data/raw/timing_repetitions.csv')
    expected_timing=checked*int(timings.timing_repetition.nunique()) if len(timings) else 0
    report={
      'records_checked':checked,
      'mismatches':mismatches,
      'envelope_failures':envelope_failures,
      'timing_rows':int(len(timings)),
      'timing_repetitions_per_record':int(timings.timing_repetition.nunique()) if len(timings) else 0,
      'expected_timing_rows':expected_timing,
      'timing_rows_complete':bool(len(timings)==expected_timing),
      'passed':bool(not mismatches and not envelope_failures and len(timings)==expected_timing),
      'method':'Exact linear error-propagation matrices, independent of iterative experiment routines.'
    }
    (root/'verification/independent_heldout-integrity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('mismatches','envelope_failures')},indent=2))
    if not report['passed']:
        raise SystemExit(1)

if __name__=='__main__': main()

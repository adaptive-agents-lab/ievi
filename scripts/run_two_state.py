"""Run the two-state illustration and export modal amplitudes and relative value errors."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

def local_path(path):
    """Keep deeply nested reference sources accessible on Windows."""
    resolved=Path(path).resolve()
    if os.name=='nt' and not str(resolved).startswith('\\\\?\\'):
        value=str(resolved)
        return Path('\\\\?\\UNC\\'+value[2:] if value.startswith('\\\\') else '\\\\?\\'+value)
    return resolved

ROOT=local_path(Path(__file__).parents[1])
SOURCE=ROOT/'methods/planning/implementation/experiments/paper_figures/two_state_bellman_flow_stiffness'
def write(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main(output):
    output=local_path(output)
    if output.exists():raise ValueError('Refusing existing study directory')
    package=output/'package';package.mkdir(parents=True);(package/'data').mkdir()
    entry=next(x for x in json.loads((ROOT/'configs/planning/catalog.json').read_text())['studies'] if x['id']=='two-state-stiffness')
    write(output/'study-provenance.json',{**entry,'adapter_sha256':sha(Path(__file__)),'observable_definition':'Slow modal amplitudes are computed explicitly; plotted relative L_inf value errors are exported separately.'})
    shutil.copytree(SOURCE/'code',package/'code',ignore=shutil.ignore_patterns('figures','__pycache__'))
    plot=package/'code/experiments/two-state-bellman-flow-stiffness/plot_strict_swap_inset.py'
    exporter=package/'code/export_two_state_compact_inset_data.py'
    text=exporter.read_text(encoding='utf-8')
    anchor='    threshold = float(slow_mode_panel["threshold"])'
    assert text.count(anchor)==1
    text=text.replace(anchor,'''    # Reference raw schema records slow modal amplitudes; the current plot
    # additionally reports normalized L_inf errors. Compute each explicitly.
    slow_mode_panel["vi_slow"] = gamma ** slow_mode_panel["vi_queries"]
    slow_mode_panel["exact_slow"] = exact_modes[:, 0]
    slow_mode_panel["surrogate_slow"] = surrogate_modes[:, 0]
'''+anchor)
    exporter.write_text(text,encoding='utf-8')
    env=dict(os.environ,MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable,str(plot)],cwd=package,env=env,check=True)
    subprocess.run([sys.executable,str(exporter)],cwd=package,env=env,check=True)
    raw=package/'data/two_state_compact_inset.raw_data.json'
    recomputed=json.loads(raw.read_text());reference=json.loads((SOURCE/'data'/raw.name).read_text())
    differences=[]
    def compare(a,b,path=''):
        if isinstance(a,dict) and isinstance(b,dict):
            for k in sorted(set(a)|set(b)):
                if k not in a or k not in b:differences.append(path+'/'+k)
                else:compare(a[k],b[k],path+'/'+k)
        elif isinstance(a,list) and isinstance(b,list):
            if len(a)!=len(b):differences.append(path+'/length')
            else:
                for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
        elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
            if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12):differences.append(path)
        elif a!=b:differences.append(path)
    compare(recomputed,reference)
    # Independent value-space recurrence checks the actual plotted quantity.
    gamma=.99;h=1/(1-gamma);p=np.array([[0.,1.],[1.,0.]]);ph=np.array([[.6,.4],[1.,0.]]);reward=np.array([1.,0.]);vstar=np.linalg.solve(np.eye(2)-gamma*p,reward)
    def recurrence(surrogate,inner_steps):
        v=np.zeros(2);errors=[]
        for k in range(15):
            errors.append(float(np.max(np.abs(v-vstar))/np.max(np.abs(vstar))))
            anchor=v.copy()
            for j in range(inner_steps):
                rhs=anchor+h*(reward+gamma*(p-surrogate)@v)
                v=np.linalg.solve((1+h)*np.eye(2)-h*gamma*surrogate,rhs)
        return np.array(errors)
    exact=recurrence(p,1);surrogate=recurrence(ph,3)
    metadata=json.loads((plot.parent/'figures/two_state_compact_inset.metadata.json').read_text())
    actual=metadata['slow_mode_panel']
    assert np.allclose(exact,actual['exact_relative_error'],rtol=1e-10,atol=1e-12)
    assert np.allclose(surrogate,actual['surrogate_relative_error'],rtol=1e-10,atol=1e-12)
    vi_cross=math.ceil(math.log(.001)/math.log(gamma));exact_cross=int(np.flatnonzero(exact<=.001)[0]);surrogate_cross=3*int(np.flatnonzero(surrogate<=.001)[0])
    assert (vi_cross,exact_cross,surrogate_cross)==(688,10,30)
    assert gamma**(vi_cross-1)>.001 and gamma**vi_cross<=.001
    write(package/'data/actual-plotted-relative-error.json',actual)
    checks={'reference_raw_schema_numerical_equivalence':not differences,'raw_differences':differences,'independent_recurrence_matches_actual_plot':True,'threshold_query_counts':{'vi':vi_cross,'exact_ievi':exact_cross,'surrogate_ievi':surrogate_cross},'plot_quantity':'relative L_inf value error','reference_raw_quantity':'slow modal amplitudes, separately computed; not aliased to relative error','source_plot_sha256':sha(SOURCE/'code/experiments/two-state-bellman-flow-stiffness/plot_strict_swap_inset.py'),'source_exporter_sha256':sha(SOURCE/'code/export_two_state_compact_inset_data.py'),'staged_exporter_sha256':sha(exporter),'raw_sha256':sha(raw),'synthetic_vi_cone_is_illustrative':True}
    write(output/'two-state-checks.json',checks)
    if differences:raise ValueError(f'Reference raw mismatch: {differences[:10]}')
    write(output/'completed.json',{'id':'two-state-stiffness','completed':True,'observable_definition':'Slow modal amplitudes and plotted relative L_inf value errors are distinct quantities; the latter are checked by an independent recurrence.','checks':'two-state-checks.json'})
    print(json.dumps(checks,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();main(a.output_root)

"""Draw current reference figures exclusively from verified recomputed solver outputs.

Supplied files define the drawing code, geometry and experimental grids.
Plotted measurements come from the verified solver outputs. Missing runs
block the affected figure; generated products use a separate output directory.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reproduction_manifest import guard_output_path, identity, load_tasks, verified_record

def long(path):
    path=Path(path).resolve()
    return Path('\\\\?\\'+str(path)) if os.name=='nt' and not str(path).startswith('\\\\?\\') else path

PLANNING_ROOT=long(ROOT/'methods/planning')
IMPLEMENTATION_ROOT=PLANNING_ROOT/'implementation'
RESTORATION_ROOT=ROOT/'methods/incumbent-restoration'
FIGURE_TASKS=['main',*[f'appendix{i:02}' for i in range(13)],'stiffness']

def read(path):return json.loads(long(path).read_text(encoding='utf-8-sig'))
def write(path,obj):
    path=long(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=True)+'\n',encoding='utf-8')
def sha(path):
    h=hashlib.sha256()
    with long(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def record(path):return {'path':str(path),'sha256':sha(path),'bytes':long(path).stat().st_size}
def module(path,name='recomputed_figure'):
    path=long(path);sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def find_run(run_root,ident,tasks):
    if ident not in tasks:return None
    fingerprint,_=identity(tasks[ident],ROOT)
    for path in sorted((run_root/ident).glob('*/execution.json'),reverse=True):
        if verified_record(path,fingerprint):return path.parent/'study'
    return None

def hit(queries,values,tau=.001,stable=True):
    valid=[math.isfinite(float(x)) and float(x)<tau for x in values]
    if stable:
        bad=[i for i,x in enumerate(valid) if not x];i=bad[-1]+1 if bad else 0
    else:
        good=[i for i,x in enumerate(valid) if x];i=good[0] if good else len(queries)
    return int(queries[i]) if i<len(queries) else None

def hit_rows(path,adaptive='waterfall_besp'):
    rows=[]
    for item in read(path):
        row={'lambda':item['lambda'],'seed':item.get('seed')}
        for public,key in [('vi','vi'),('osvi','osvi'),('aievi',adaptive)]:
            t=item['traces'][key];q=t['queries'];errors=t.get('errors',t.get('policy_value_errors'))
            row.update({f'{public}_stable_hit':hit(q,errors),f'{public}_first_hit':hit(q,errors,stable=False),f'{public}_last_query':q[-1],f'{public}_last_error':errors[-1],f'{public}_max_error':max(errors)})
        rows.append(row)
    return rows

def copy_drawing_sources(destination):
    """Whitelist only code and geometric figure specs: no archived measurements."""
    copied=[]
    for path in IMPLEMENTATION_ROOT.rglob('*'):
        if not path.is_file() or not (path.suffix=='.py' or path.name=='figure_spec.json'):continue
        target=destination/path.relative_to(IMPLEMENTATION_ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target);copied.append(record(path))
    return copied

def assemble(base,adaptive,grid,output,aliases=None):
    """Merge recomputed independent baseline/adaptive streams by exact lambda/seed."""
    indexed={float(x['lambda']):copy.deepcopy(x) for x in read(base)}
    for value in grid:
        if float(value) not in indexed:raise ValueError(f'Recomputed baseline missing lambda={value}: {base}')
    for path,key in adaptive:
        incoming={float(x['lambda']):x for x in read(path)}
        for value in grid:
            row=indexed[float(value)];other=incoming.get(float(value))
            if other is None:raise ValueError(f'Recomputed adaptive trace missing lambda={value}: {path}')
            if other.get('seed')!=row.get('seed'):raise ValueError(f'Baseline/adaptive seed mismatch at {value}')
            trace_keys=list(other['traces'])
            if len(trace_keys)!=1:raise ValueError(f'Expected one adaptive trace in {path}: {trace_keys}')
            row['traces'][key]=copy.deepcopy(other['traces'][trace_keys[0]])
            row.setdefault('results',{})[key]=copy.deepcopy(other.get('results',{}).get(trace_keys[0],{}))
    selected=[indexed[float(x)] for x in grid]
    if aliases:
        for row in selected:
            for dest,source in aliases.items():row['traces'][dest]=copy.deepcopy(row['traces'][source])
    write(output,selected)
    return selected

def worker(task,context_path):
    c=read(context_path);orig=Path(c['drawing_root']);packages=orig/'experiments/paper_figures';figs=Path(c['figures']);tables=Path(c['tables']);paths=[Path(x) for x in c['trace_paths']]
    sys.path.insert(0,str(orig/'experiments'));sys.path.insert(0,str(orig))
    png=figs/(c['asset']+'.png')
    if task=='main':
        from scripts.render_main_two_panel import planning_metadata, render_figure
        cfg=read(c['planning_config'])
        planning=planning_metadata(hit_rows(paths[0],adaptive='waterfall'),cfg['max_queries'],c['planning_tasks'])
        metadata=render_figure(figs/c['asset'], planning=planning)
        metadata['provenance']=c['main_provenance']
        metadata['outputs']={suffix:record(figs/(c['asset']+'.'+suffix)) for suffix in ['pdf','png','data.npz']}
        write(figs/(c['asset']+'.metadata.json'),metadata)
    else:
        i=int(task[-2:]);package=next(packages.glob(f'appendix_figure_{i:02}_*'))
        if i==0:
            (package/'README.md').write_text('Recomputed schematic rendering; geometry from reference specification; no measured curves.\n')
            m=module(package/'code/render_environment_schematics.py');m.render()
            for p in (orig/'figures').glob('environment_schematics.*'):shutil.copy2(p,figs/p.name)
        elif i in (1,2,11):
            script=next((package/'code/experiments/scripts').glob('render_*gamma_comparison*.py'));m=module(script)
            spec=read(package/'figure_spec.json')
            for dataset,path in zip(spec['datasets'],paths,strict=True):dataset['trace_file']=str(path)
            write(package/'figure_spec.json',spec);m.render(package/'figure_spec.json',png)
        elif i in (5,6,8):
            script={5:'experiments/scripts/render_threshold_criterion_sensitivity.py',6:'experiments/scripts/render_aievi_schedule_sensitivity.py',8:'render_maze_control_self_loop_gamma099.py'}[i]
            m=module(package/'code'/script)
            if i==6:m.render(paths[0],png,tables/'schedule-sensitivity')
            else:m.render(paths[0],png)
        elif i==10:
            m=module(package/'code/render_cliffwalk_pe_self_loop_grid.py')
            m.load_rows=lambda path,**kwargs:hit_rows(path)
            m.render(*paths,png)
        else:
            script=next((package/'code').glob('render_*gamma_comparison.py'));m=module(script);m.render(*paths,png)

def main(run_root,output,selected,accepted_dataset=None):
    guard_output_path(output,ROOT)
    run_root=run_root.resolve();output=long(output)
    if output.exists():raise ValueError('Choose a new nonexistent derivative output directory')
    output.mkdir(parents=True);figs=output/'figures';figs.mkdir();(output/'logs').mkdir();(output/'tables').mkdir()
    tasks=load_tasks(ROOT);catalog=read(ROOT/'configs/planning/selected/catalog.json')['studies'];catalog=[x for x in catalog if x['id']!='selected-evaluation-cliffwalk-self-loop-gamma099-boundary']
    configs={p.stem:read(p) for p in (RESTORATION_ROOT/'configs').glob('*.json')}
    requirements={f'appendix{i:02}':[] for i in range(13)}
    bindings={}
    for stem,cfg in configs.items():
        number=int(stem[:2]);requirements[f'appendix{number:02}'].append('planning-'+stem)
        normalized=(cfg.get('source_config') or '').replace('\\','/').replace('runner_config_snapshot.yaml','source_config.yaml')
        matches=[x for x in catalog if any(s['path'].endswith('/'+normalized) for s in x['source_configs'])]
        if number in (5,6):matches=[x for x in catalog if x['id']=='selected-control-cliffwalk-gamma099']
        if len(matches)!=1:raise ValueError(f'Cannot uniquely bind recomputed baseline for {stem}: {[x["id"] for x in matches]}')
        bindings[stem]=matches[0];requirements[f'appendix{number:02}'].append(matches[0]['id'])
    for number,ids in [(7,['selected-control-maze-gamma09','selected-control-maze-gamma099']),(8,['selected-control-maze-self-loop-gamma099'])]:requirements[f'appendix{number:02}']=ids
    requirements['stiffness']=['two-state-stiffness']
    requirements['main']=['planning-05_threshold','selected-control-cliffwalk-gamma099']
    required=sorted({ident for figure in selected for ident in requirements[figure]})
    if accepted_dataset:
        from scripts.reference_dataset import resolve_attempt
        runs={ident:resolve_attempt(accepted_dataset,ident)[0].parent/'study' for ident in required}
    else:
        runs={ident:find_run(run_root,ident,tasks) for ident in required}
    missing=[ident for ident,path in runs.items() if path is None]
    sources={};assembly=[];model_checks=[];results=[];drawing=output/'drawing-source';copied=copy_drawing_sources(drawing)
    assets={int(Path(x['config_path']).name.split('_')[2]):Path(x['output_file']).stem for x in csv.DictReader((PLANNING_ROOT/'FIGURE-PROVENANCE.csv').open()) if '/appendix_figure_' in x['config_path']}
    def raw(ident):
        study=runs[ident]
        if ident.startswith('planning-'):path=study/'solver/raw-traces.json'
        else:
            choices=list((study/'solver/raw').glob('*_traces.json'))
            if len(choices)!=1:raise ValueError(f'Expected one recomputed trajectory file: {ident}: {choices}')
            path=choices[0]
        sources.setdefault(ident,[]).append(record(path));return path
    def check_model(stem):
        import numpy as np
        base_model=runs[bindings[stem]['id']]/'inputs/environment-target.npz'
        adaptive_model=runs['planning-'+stem]/'solver/environment-target.npz'
        with np.load(long(base_model)) as b,np.load(long(adaptive_model)) as a:
            exact=all(np.array_equal(b[k],a[k]) for k in ['p','r','gamma'])
            targets=all(np.allclose(b[k],a[k],rtol=1e-10,atol=1e-8) for k in ['v_pi_star','v_star_control'])
        if not exact or not targets:raise ValueError(f'Recomputed baseline/adaptive true-model or target mismatch: {stem}')
        sources.setdefault(bindings[stem]['id'],[]).append(record(base_model));sources.setdefault('planning-'+stem,[]).append(record(adaptive_model))
        model_checks.append({'run_id':stem,'true_transition_reward_gamma_exact':exact,'target_arrays_match_1e-10_1e-8':targets})
    def merge_stem(stem,path):
        cfg=configs[stem];base=raw(bindings[stem]['id']);adaptive=raw('planning-'+stem)
        check_model(stem)
        key='waterfall_besp' if cfg['problem_type']=='pe' else 'waterfall'
        rows=assemble(base,[(adaptive,key)],cfg['lambdas'],path)
        assembly.append({'output':record(path),'baseline_task':bindings[stem]['id'],'adaptive_task':'planning-'+stem,'grid':cfg['lambdas'],'series':list(rows[0]['traces'])});return path
    for figure in selected:
        absent=[x for x in requirements[figure] if runs[x] is None]
        if absent:results.append({'figure':figure,'status':'blocked_missing_recomputed_runs','missing':absent});continue
        context={'drawing_root':str(drawing),'figures':str(figs),'tables':str(output/'tables'),'trace_paths':[]}
        if figure=='stiffness':
            study=runs['two-state-stiffness'];folder=study/'package/code/experiments/two-state-bellman-flow-stiffness/figures'
            # Re-render analytical illustration from the verified plotting implementation.
            plot=folder.parent/'plot_strict_swap_inset.py'
            staged=output/'analytic-source';shutil.copytree(long(study/'package/code'),staged,ignore=shutil.ignore_patterns('figures','__pycache__'))
            staged_folder=staged/'experiments/two-state-bellman-flow-stiffness'
            sources.setdefault('two-state-stiffness',[]).append(record(plot))
            subprocess.run([sys.executable,str(staged_folder/plot.name)],cwd=staged,env=dict(os.environ,MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1'),check=True)
            for path in (staged_folder/'figures').glob('two_state_compact_inset.*'):
                shutil.copy2(long(path),figs/path.name)
            source=study/'package/data/two_state_compact_inset.raw_data.json';sources['two-state-stiffness'].append(record(source));results.append({'figure':figure,'status':'completed','asset':'two_state_compact_inset.pdf','kind':'recomputed analytic illustration output'});continue
        if figure=='main':
            path=merge_stem('05_threshold',output/'assembled/main-planning-traces.json');context['trace_paths']=[str(path)];context['asset']='figure_main_empirical_results_composite'
            context['planning_config']=str(RESTORATION_ROOT/'configs/05_threshold.json')
            context['planning_tasks']=['selected-control-cliffwalk-gamma099','planning-05_threshold']
            context['main_provenance']={
                'dataset':record(accepted_dataset) if accepted_dataset else None,
                'selected_attempts':{ident:{'attempt':runs[ident].parent.name,'execution':record(runs[ident].parent/('DATA.json' if accepted_dataset else 'execution.json'))} for ident in requirements['main']},
                'source_files':{ident:sources[ident] for ident in requirements['main']},
                'planning_config':record(context['planning_config']),
                'generator_files':[record(ROOT/'scripts/render_main_two_panel.py'),record(Path(__file__))],
                'solver_experiments_run':False,'archived_measurement_fallback':False}
        else:
            number=int(figure[-2:]);context['asset']=assets[number]
            if number in (7,8):
                for ident in requirements[figure]:
                    entry=next(x for x in catalog if x['id']==ident);path=output/'assembled'/figure/(ident+'.json')
                    assemble(raw(ident),[],entry['selected_grid'],path);context['trace_paths'].append(str(path));assembly.append({'output':record(path),'task':ident,'series':['vi','osvi','aisp'],'grid':entry['selected_grid']})
            elif number==6:
                stems=sorted(x for x in configs if x.startswith('06_'));grid=configs[stems[0]]['lambdas'];incoming=[]
                for stem in stems:
                    check_model(stem)
                    a=configs[stem]['algorithms'][0];name=f"waterfall_d{a['decay_factor']:.2f}_s{a['stall_patience']}".replace('.','p');incoming.append((raw('planning-'+stem),name))
                path=output/'assembled/schedule-traces.json';rows=assemble(raw('selected-control-cliffwalk-gamma099'),incoming,grid,path);context['trace_paths']=[str(path)];assembly.append({'output':record(path),'tasks':requirements[figure],'series':list(rows[0]['traces']),'grid':grid})
            elif number!=0:
                stems=sorted(x for x in configs if x.startswith(f'{number:02}_'))
                if number==10:stems=['10_gamma099_budget1000','10_gamma099_budget10000','10_gamma0999_budget1000','10_gamma0999_budget10000']
                for stem in stems:context['trace_paths'].append(str(merge_stem(stem,output/'assembled'/figure/(stem+'.json'))))
        context_path=output/'contexts'/(figure+'.json');write(context_path,context)
        cmd=[sys.executable,str(long(__file__)),'--worker',figure,'--context',str(context_path)]
        with (output/'logs'/(figure+'.log')).open('w',encoding='utf-8') as f:completed=subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,MPLBACKEND='Agg',PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT)
        passed=completed.returncode==0 and (figs/(context['asset']+'.pdf')).exists()
        results.append({'figure':figure,'status':'completed' if passed else 'failed','exit_code':completed.returncode,'asset':context['asset']+'.pdf','context':record(context_path),'command':cmd});print(figure,results[-1]['status'],flush=True)
    for result in results:result['source_tasks']=requirements[result['figure']]
    run_provenance={ident:{'study':str(path),'execution':record(path.parent/('DATA.json' if accepted_dataset else 'execution.json'))} for ident,path in runs.items() if path is not None}
    source_preservation=all(sha(x['path'])==x['sha256'] for records in sources.values() for x in records)
    report={'scope':'Planning figures drawn from verified measured streams; reference solver build identity not claimed','label':'Planning results','status':'complete' if all(x['status']=='completed' for x in results) and source_preservation else 'incomplete','missing_recomputed_tasks':missing,'figures':results,'recomputed_sources':sources,'source_files_unchanged':source_preservation,'accepted_dataset':str(accepted_dataset) if accepted_dataset else None,'verified_attempts':run_provenance,'assembled_inputs':assembly,'recomputed_model_compatibility':model_checks,'drawing_sources':copied,'builder':record(Path(__file__)),'active_main_renderer':record(ROOT/'scripts/render_main_two_panel.py'),'archived_measurement_fallback':False,'outputs':[record(p) for p in figs.glob('*.pdf')],'qualifications':['Independent target reconstruction; consult each solver attempt certificate','Schematic geometry is a reference model specification, not a measured stream','All numerical mismatches remain new results; no archived curve substitution']}
    write(output/'planning-render-report.json',report)
    (output/'README.md').write_text('# Planning figures\n\nMeasured series are computed from verified scientific data. See planning-render-report.json for source hashes and coverage.\n',encoding='utf-8')
    return report['status']=='complete'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,default=ROOT/'runs/recomputed');p.add_argument('--dataset',type=Path);p.add_argument('--output-root',type=Path);p.add_argument('--figures',nargs='+',choices=FIGURE_TASKS);p.add_argument('--worker',choices=FIGURE_TASKS);p.add_argument('--context',type=Path);a=p.parse_args()
    if a.worker:worker(a.worker,a.context)
    elif a.output_root is None:p.error('--output-root is required')
    else:sys.exit(0 if main(a.run_root,a.output_root,a.figures or FIGURE_TASKS,a.dataset) else 3)

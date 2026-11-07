"""Generate current paper artifacts from verified data; no solvers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reproduction_manifest import sha256, utc_now, write_json
from scripts.paper_tables import generate


def render(dataset: Path, output: Path):
    dataset, output = dataset.resolve(), output.resolve()
    if output.exists():
        raise ValueError('Choose a new nonexistent output directory; existing artifacts are preserved')
    # Even caller-selected destinations may never enter the manuscript or frozen inputs.
    from scripts.reproduction_manifest import guard_output_path
    guard_output_path(output, ROOT)
    output.mkdir(parents=True)
    work = ROOT / '.cache/paper-render' / uuid.uuid4().hex
    work.mkdir(parents=True)
    commands = []
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
    for script, destination in [('render_planning.py', 'reference'), ('build_planning.py', 'final')]:
        command = [sys.executable, str(ROOT / 'scripts' / script), '--dataset', str(dataset), '--output-root', str(work / destination)]
        with (output / (destination + '-render.log')).open('wb') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        commands.append({'argv':command, 'exit_code':result.returncode, 'log':destination+'-render.log'})
        if result.returncode:
            raise ValueError(f'{destination} rendering failed; inspect {output / (destination+"-render.log")}')
    return export_prepared(dataset, output, work, commands)


def export_prepared(dataset, output, work, commands):
    """Export verified figure and summary products to the selected output directory."""
    planning = json.loads((work / 'reference/planning-render-report.json').read_text())
    final = json.loads((work / 'final/planning-build-report.json').read_text())
    if planning['status']!='complete' or not planning['source_files_unchanged']:
        raise ValueError('Reference source integrity gate failed')
    if final['status']!='completed' or not final['source_attempts_still_verified'] or not final['renderer_code_unchanged']:
        raise ValueError('Final source integrity gate failed')
    tables = generate(dataset, output / 'tables')
    if tables['status'] != 'completed' or not tables['all_generated_cells_match_aggregation']:
        raise ValueError('Empirical table reconciliation did not pass')
    coverage = json.loads((ROOT / 'docs/coverage/coverage.json').read_text())
    current_stem = work / 'reference/figures/figure_main_empirical_results_composite'
    current_main = json.loads(current_stem.with_suffix('.metadata.json').read_text())
    if not current_main['pixel_identical_to_reference_at_300dpi'] or not current_main['planning_native_content_only']:
        raise ValueError('Current Figure 1 verification failed')
    assets = []
    for entry in coverage['figure_inventory']:
        paper_path = entry['asset']
        final_asset = Path(paper_path).name.startswith(('E1-', 'E2-', 'E3-', 'E4-'))
        source = (work / 'final/derived/figures' if final_asset else work / 'reference/figures') / Path(paper_path).name
        is_current_main = source.name == 'figure_main_empirical_results_composite.pdf'
        target = output / 'figures' / paper_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if source.with_suffix('.png').exists():
            shutil.copy2(source.with_suffix('.png'), target.with_suffix('.png'))
        if source.with_suffix('.metadata.json').exists():
            shutil.copy2(source.with_suffix('.metadata.json'), target.with_suffix('.metadata.json'))
        if is_current_main:
            shutil.copy2(source.with_suffix('.data.npz'), target.with_suffix('.data.npz'))
        if final_asset:
            task_ids = {'E1':'planning-control','E2':'planning-control','E3':'planning-horizon','E4':'planning-timing'}
            task = task_ids[target.name[:2]]
            inputs = {task: final['source_attempts'][task]}
        else:
            # Context and full assembled-input map preserve exact stream membership.
            item = next(x for x in planning['figures'] if x['asset'] == target.name)
            inputs = {task: {'execution':planning['verified_attempts'][task]['execution'], 'files':planning['recomputed_sources'].get(task,[])} for task in item['source_tasks']}
            if not inputs: inputs={'schematic-geometry':planning['drawing_sources']}
        if is_current_main:
            inputs = {'planning-reference': inputs,
                      'cliffwalk': current_main['source_data']}
        assets.append({'paper_path':paper_path, 'candidate_path':target.relative_to(output).as_posix(),
                       'sha256':sha256(target),'source_tasks':list(inputs),'inputs':inputs,
                       'generator': ('scripts/build_planning.py' if final_asset else 'scripts/render_main_two_panel.py' if target.name=='figure_main_empirical_results_composite.pdf' else 'scripts/render_planning.py')})
    provenance = output / 'provenance'
    provenance.mkdir(exist_ok=True)
    for source in [work/'final/inputs/hardware-and-protocol.json', work/'reference/planning-render-report.json', work/'final/planning-build-report.json', work/'final/recomputed-statistics-staging.json']:
        shutil.copy2(source, provenance / source.name)
    for name in ['EXPERIMENT-PROTOCOL.md','EXPERIMENT-EVIDENCE.md','BUILD-CHECKS.json']:
        shutil.copy2(work/'final/derived'/name, output/name)
        if name.endswith('.md'):
            text=(output/name).read_text(encoding='utf-8')
            text=text.replace('`inputs/`','the verified task data').replace('inputs/', 'task output: ')
            text+='\n\n## Data sources\n\nTask planning-control supplies control_accuracy/raw/control_accuracy_summary.csv; planning-horizon supplies horizon_scaling/data/raw/records.csv and horizon summaries; planning-timing supplies wallclock/data/raw/benchmark_runs.csv. Data files are verified against their scientific record. The method comparison joins 57 primary baseline rows with 38 incumbent restoration rows.\n'
            (output/name).write_text(text,encoding='utf-8')
    shutil.copytree(work/'final/derived/data',output/'data')
    exported_tables=[]
    for x in tables['generated_tables']:
        blocks=[{'path':'tables/'+x['path'],'sha256':x['sha256']}] if x['blocks']==1 else [
            {'path':'tables/'+Path(x['path']).stem+f'-block-{i}.tex',
             'sha256':sha256(output/'tables'/(Path(x['path']).stem+f'-block-{i}.tex'))} for i in range(x['blocks'])]
        source_tasks=['planning-control','restoration-common'] if x['label']=='tab:incumbent-restoration-comparison' else ['planning-horizon'] if 'horizon' in x['label'] else ['planning-timing'] if 'runtime' in x['label'] else ['planning-control']
        exported_tables.append(dict(x,path='tables/'+x['path'],block_count=x['blocks'],blocks=blocks,source_tasks=source_tasks,
                                    inputs=[source for source in tables['recomputed_sources'] if source['task'] in source_tasks]))
    assert len(assets)==19 and len(exported_tables)==11 and tables['generated_cells']==628
    manifest = {'schema_version':1,'status':'completed','generated_at':utc_now(),
                'dataset':{'path':str(dataset.relative_to(ROOT)) if dataset.is_relative_to(ROOT) else str(dataset),'sha256':sha256(dataset)},
                'main_panel_version':'paper',
                'dataset_layers':['planning-reference: planning panel, 18 other figures, 11 tables',
                                  'cliffwalk: raw-selected current RL panel'],
                'current_main_validation':current_main,
                'producing_command':sys.argv,'generator_sha256':sha256(Path(__file__)),
                'generator_files_sha256':{name:sha256(ROOT/name) for name in ['scripts/render_paper.py','scripts/paper_tables.py','scripts/build_planning.py','scripts/render_planning.py','scripts/render_main_two_panel.py','scripts/reference_dataset.py', 'scripts/cliffwalk.py']},
                'solver_experiments_run':False,'archived_measurement_fallback':False,
                'figure_count':len(assets),'empirical_table_count':len(exported_tables),'table_cell_count':628,
                'cell_comparison_counts':tables['counts'],'all_generated_cells_match_aggregation':True,
                'restoration_baseline_origin':'57 baseline rows from accepted planning-control; 38 adaptive rows from accepted restoration-common',
                'assets':assets,'tables':exported_tables,'commands':commands,
                'files':{p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file()}}
    main_description = 'Figure 1 combines the planning panel with Cliffwalk raw trajectories. '
    (output/'README.md').write_text('Paper figures and tables from verified scientific data.\n\n19 assets and 11 empirical tables (628 compared data cells). '+main_description+'Every table value is recomputed from selected raw data. E1/E2 use the primary algorithm; incumbent restoration is a separate method comparison. See MANIFEST.json and the experiment protocol for data sources.\n',encoding='utf-8')
    manifest['files']['README.md']=sha256(output/'README.md')
    write_json(output/'MANIFEST.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'datasets/reference/planning/DATASET.json')
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    manifest=render(args.dataset,args.output_root)
    print(json.dumps({k:manifest[k] for k in ['status','figure_count','empirical_table_count','table_cell_count','cell_comparison_counts']},indent=2))


if __name__=='__main__':
    main()

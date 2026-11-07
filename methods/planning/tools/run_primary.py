from pathlib import Path
import subprocess,sys,os,json,time,platform,importlib.metadata,argparse
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description='Run control, horizon, and timing studies into an empty directory; existing results are protected.')
p.add_argument('--output-root',type=Path,required=True)
args=p.parse_args();out=args.output_root.resolve();env=os.environ.copy()
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS']: env[k]='1'
env['JAX_PLATFORMS']='cpu'
if out.exists() and any(out.iterdir()): raise SystemExit('Refusing to overwrite nonempty output; choose a new empty directory with --output-root.')
out.mkdir(parents=True,exist_ok=True)
meta={'cpu':platform.processor() or None,'sockets':None,'physical_cores':None,'logical_processors':os.cpu_count(),'installed_ram_GiB':None,'os_visible_ram_bytes':None,'os':platform.platform(),'gpu_used':False,'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','numba','matplotlib']},'threads_per_process':1,'control_accuracy_workers':4,'horizon_scaling_seeds':list(range(202609170000,202609170030)),'wallclock_repetitions':10,'started':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
probe="$c=@(Get-CimInstance Win32_Processor); $m=Get-CimInstance Win32_ComputerSystem; $o=Get-CimInstance Win32_OperatingSystem; $ram=(Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum).Sum; @{cpu=($c.Name -join '; '); sockets=$c.Count; physical_cores=($c | Measure-Object NumberOfCores -Sum).Sum; logical_processors=($c | Measure-Object NumberOfLogicalProcessors -Sum).Sum; installed_ram_GiB=$ram/1GB; os_visible_ram_bytes=$m.TotalPhysicalMemory; os=($o.Caption+' '+$o.Version)} | ConvertTo-Json"
if platform.system()=='Windows':
    meta.update(json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',probe],text=True)))
else:
    meta['hardware_probe']='Platform metadata; unmeasured RAM/core/socket fields remain null.' 
(out/'hardware-and-protocol.json').write_text(json.dumps(meta,indent=2))
jobs=[('control_accuracy',['--output-dir',str(out/'control_accuracy'),'--workers','4']),('horizon_scaling',['--output-root',str(out/'horizon_scaling'),'--replications','30','--base-seed','202609170000','--timing-repetitions','5']),('wallclock_10k',['--output-dir',str(out/'wallclock'),'--reps','10'])]
for name,args in jobs:
 cmd=[sys.executable,'-u',str(R/'benchmarks/scripts'/{'control_accuracy':'run_control.py','horizon_scaling':'run_horizon.py','wallclock_10k':'run_timing.py'}[name]),*args]
 print('Starting',name,flush=True)
 with (out/f'{name}.log').open('w') as f: ret=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 print('Finished',name,ret,flush=True)
 if ret: raise SystemExit(ret)
meta['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z'); (out/'hardware-and-protocol.json').write_text(json.dumps(meta,indent=2))

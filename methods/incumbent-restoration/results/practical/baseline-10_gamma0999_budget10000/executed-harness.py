"""Practical-family measurements using the incumbent-restoration dispatcher and explicit inputs.

Only the inner Control VI loop is compiled; its update/stopping semantics match
the Python loop. All practical outer state transitions use the
dispatcher verbatim. Source target files are unavailable; targets are certified independently.
"""
from pathlib import Path
import os, sys, hashlib
EXECUTED_HARNESS=Path(__file__).read_bytes()
EXECUTED_HARNESS_SHA256=hashlib.sha256(EXECUTED_HARNESS).hexdigest()
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['JAX_ENABLE_X64']='true'
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
REV=Path(__file__).resolve().parents[1]
SOURCE=REV.parents[1]/'methods/planning/implementation/src' if '--primary' in sys.argv else REV/'implementation/src'
sys.path.insert(0,str(SOURCE))
import argparse, hashlib, importlib.metadata, inspect, json, platform, time
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from besp.runners import deterministic_dispatch as dd
from besp.runners.deterministic_runtime import LoadedDeterministicTarget,compute_mdp_fingerprint
from besp.environments.jax_native import create_environment_jax_native
from besp.model_error_runtime import apply_model_error_for_task

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=True),encoding='utf-8')

@jax.jit
def compiled_inner(v,p_hat,r_tilde,gamma_eff,tols,caps):
    def cond(state):
        i,_,proceed,_,_=state
        return (i<jnp.max(caps)) & proceed
    def body(state):
        i,v,_,last_delta,updates=state
        q=r_tilde+gamma_eff[:,None,None]*jnp.einsum('bsak,bk->bsa',p_hat,v)
        u=jnp.max(q,axis=2)
        delta=jnp.max(jnp.abs(u-v),axis=1)
        change=(i<caps)&(delta>=tols)
        return (i+1,jnp.where(change[:,None],u,v),jnp.any(change),delta,updates+change.astype(jnp.int32))
    return jax.lax.while_loop(cond,body,(jnp.int32(0),v,jnp.bool_(True),jnp.full(v.shape[0],jnp.inf),jnp.zeros(v.shape[0],jnp.int32)))

inner_diagnostics=[]
outer_diagnostics=[]
outer_counts=None
def observe_outer(active,failed,restore,charged_queries,h,previous_h):
    global outer_counts
    a=np.asarray(active,dtype=bool);f=np.asarray(failed,dtype=bool);restore=np.asarray(restore,dtype=bool)
    if outer_counts is None:
        outer_counts={'active_attempts':np.zeros(len(a),dtype=int),'candidate_true_products_all_lanes':np.zeros(len(a),dtype=int),'guard_events':np.zeros(len(a),dtype=int),'restoration_events':np.zeros(len(a),dtype=int)}
    outer_counts['active_attempts']+=a
    outer_counts['candidate_true_products_all_lanes']+=1
    outer_counts['guard_events']+=f
    outer_counts['restoration_events']+=restore
    if np.any(f|restore):
        outer_diagnostics.append({'active':a.tolist(),'failed':f.tolist(),'restored':restore.tolist(),'charged_queries':np.asarray(charged_queries).tolist(),'previous_h':np.asarray(previous_h).tolist(),'next_h':np.asarray(h).tolist()})

def instrument_outer(regime,old=False):
    name='_batched_waterfall_pe_common' if regime=='pe' else '_batched_waterfall_control_traces'
    source=inspect.getsource(getattr(dd,name))
    marker='        restore = jnp.logical_and(active, h < previous_h)\n'
    instrumented=source
    if not old:
        assert source.count(marker)==1
        instrumented=source.replace(marker,marker+'        _restoration_observe(active, failed, restore, charged_queries, h, previous_h)\n')
    collector_change=False
    if regime=='pe':
        slow='''        for local_index in range(batch_size):
            if bool(active[local_index]):
                errors = errors.at[local_index, int(charged_queries[local_index])].set(
                    current_errors[local_index]
                )
'''
        fast='''        rows_for_recording = jnp.arange(batch_size)
        errors = errors.at[rows_for_recording, charged_queries].set(
            jnp.where(active, current_errors, errors[rows_for_recording, charged_queries])
        )
'''
        assert instrumented.count(slow)==1
        instrumented=instrumented.replace(slow,fast)
        collector_change=True
    dd.__dict__['_restoration_observe']=observe_outer
    exec(compile(instrumented,'<read-only-observer:'+name+'>','exec'),dd.__dict__)
    return {'function':name,'original_function_sha256':hashlib.sha256(source.encode()).hexdigest(),'instrumented_function_sha256':hashlib.sha256(instrumented.encode()).hexdigest(),'transformation':'Read-only observer after restore-mask assignment; all solver state transition statements retained verbatim. PE trace write vectorized with same indices and values.' if not old else 'Retained outer solver with equivalent vectorized PE trace write.','vectorized_pe_collector':collector_change}
def fast_inner(v,p_hat,r_tilde,gamma_eff,tols,caps):
    iterations,value,_,delta,updates=compiled_inner(v,p_hat,r_tilde,gamma_eff,tols,caps)
    inner_diagnostics.append({'iterations':int(iterations),'updates':np.asarray(updates).tolist(),'cap_hits':np.asarray((updates>=caps)&(delta>=tols)).tolist(),'last_delta':np.asarray(delta).tolist()})
    return value

def check_inner_parity():
    rng=np.random.default_rng(9127)
    p=rng.uniform(size=(3,4,2,4));p/=p.sum(axis=-1,keepdims=True)
    v=jnp.array(rng.normal(size=(3,4)));r=jnp.array(rng.normal(size=(3,4,2)))
    g=jnp.array([.5,.9,.99]);tol=jnp.array([1e-10,1e-10,1e-10]);cap=jnp.array([100,70,10])
    old=dd._batched_waterfall_inner_vi(v,jnp.array(p),r,g,tol,cap)
    new=compiled_inner(v,jnp.array(p),r,g,tol,cap)[1]
    error=float(jnp.max(jnp.abs(old-new)))
    assert np.allclose(old,new,rtol=1e-13,atol=1e-13),(old,new,error)
    return {'fixture':'finite heterogeneous discount/cap loop, unchanged previous-iterate tolerance rule','max_absolute_difference':error,'rtol':1e-13,'atol':1e-13}

@jax.jit
def target_vi(p,r,gamma):
    def cond(s): return (s[0]<100000)&(s[2]>1e-12)
    def body(s):
        nxt=jnp.max(r+gamma*jnp.einsum('sak,k->sa',p,s[1]),axis=1)
        return s[0]+1,nxt,jnp.max(jnp.abs(nxt-s[1]))
    return jax.lax.while_loop(cond,body,(jnp.int32(0),jnp.zeros(p.shape[0]),jnp.float64(jnp.inf)))

def build_target(mdp,cfg,out):
    _,v,_=target_vi(mdp.P,mdp.R,jnp.float64(mdp.gamma))
    p=np.asarray(mdp.P);r=np.asarray(mdp.R);gamma=float(mdp.gamma)
    pi=np.asarray(jnp.argmax(mdp.R+gamma*jnp.einsum('sak,k->sa',mdp.P,v),axis=1),dtype=np.int32)
    ix=np.arange(len(pi));p_pi=p[ix,pi];r_pi=r[ix,pi]
    v_pi=np.linalg.solve(np.eye(len(pi))-gamma*p_pi,r_pi)
    resid=float(np.max(np.abs(np.max(r+gamma*np.einsum('sak,k->sa',p,v_pi),axis=1)-v_pi)))
    pe_resid=float(np.max(np.abs(r_pi+gamma*p_pi@v_pi-v_pi)))
    cert={'bellman_optimality_residual_inf':resid,'policy_evaluation_residual_inf':pe_resid,'value_error_upper_bound':resid/(1-gamma),'policy_optimal_to_float64_precision':resid/(1-gamma)<1e-7,'construction':'JAX value iteration to iterate delta<=1e-12 or cap100000, first-index argmax then numpy float64 direct policy evaluation','reference_target_recovered':False}
    assert cert['policy_optimal_to_float64_precision'],cert
    np.savez_compressed(out/'environment-target.npz',p=p,r=r,gamma=gamma,pi_star=pi,p_pi=p_pi,r_pi=r_pi,v_pi_star=v_pi,v_star_control=v_pi)
    fingerprint=compute_mdp_fingerprint(mdp)
    cert['mdp_fingerprint']=fingerprint
    cert['reference_target_path']=cfg.get('reference_target_manifest')
    save(out/'target-certification.json',cert)
    return LoadedDeterministicTarget(target_id='verified-reconstruction-'+fingerprint[:12],manifest_path=out/'target-certification.json',lane=cfg['problem_type'],env_name=cfg['env'],env_params=cfg['env_params'],gamma=gamma,mdp_fingerprint=fingerprint,pi_star=pi,v_star_control=v_pi,v_pi_star=v_pi,p_pi=p_pi,r_pi=r_pi,certification=cert)

def stable(queries,errors,threshold):
    valid=np.isfinite(errors)&(np.asarray(errors)<threshold)
    failed=np.flatnonzero(~valid)
    first=0 if len(failed)==0 else int(failed[-1])+1
    return int(queries[first]) if first<len(queries) else None
def first(queries,errors,threshold):
    hit=np.flatnonzero(np.isfinite(errors)&(np.asarray(errors)<threshold))
    return int(queries[hit[0]]) if len(hit) else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--output',required=True);ap.add_argument('--budget',type=int);ap.add_argument('--primary',action='store_true');args=ap.parse_args()
    cfg_path=REV/'configs'/f'{args.run_id}.json';cfg=json.loads(cfg_path.read_text())
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'executed-harness.py').write_bytes(EXECUTED_HARNESS)
    if args.budget:cfg['max_queries']=args.budget
    save(out/'effective-config.json',cfg)
    start=time.time()
    parity=check_inner_parity() if cfg['problem_type']=='control' else None
    if parity: dd._batched_waterfall_inner_vi=fast_inner
    instrumentation=instrument_outer(cfg['problem_type'],old=args.primary)
    params=dict(cfg['env_params']);params['seed']=0
    mdp=create_environment_jax_native(cfg['env'],**params)
    target=build_target(mdp,cfg,out)
    print('prepared',args.run_id,'states',mdp.P.shape[0],'target',target.mdp_fingerprint,flush=True)
    algs={a['name']:{k:v for k,v in a.items() if k!='name'} for a in cfg['algorithms']}
    for a in algs.values():
        for k in ['h_init','h_min','decay_factor','inner_vi_tol']:
            if k in a:a[k]=float(a[k])
    tasks=[]
    for lam,seed in zip(cfg['lambdas'],cfg['row_seeds'],strict=True):
        spec=dd.DeterministicTaskSpec(env_name=cfg['env'],env_params=params,lam=lam,seed=seed,max_queries=cfg['max_queries'],threshold=float(cfg['threshold']),algo_configs=algs,error_mode=cfg['error_mode'],problem_type=cfg['problem_type'],model_norm=cfg.get('model_norm','Linf'),value_norm='L1',value_normalized=True,emit_pe_traces=cfg['problem_type']=='pe',emit_control_traces=cfg['problem_type']=='control',metric_mode='policy_value_error' if cfg['problem_type']=='control' else 'residual_hit_time',evaluation_pipeline='online')
        p_hat=apply_model_error_for_task(mdp.P,lam=lam,mode=cfg['error_mode'],seed=seed,env_name=cfg['env'],env_params=params)
        tasks.append(dd.PreparedDeterministicTask(spec=spec,mdp=mdp,p_hat=p_hat,target_bundle=target))
    if cfg['problem_type']=='pe': outputs=dd.execute_batched_pe_tasks(tasks)
    else: outputs=dd.execute_batched_control_tasks(tasks)
    raw=[];summary=[]
    for lam,seed,row in zip(cfg['lambdas'],cfg['row_seeds'],outputs,strict=True):
        # Dataclass serialization is explicit and includes actual policy traces.
        traces=row.traces
        assert traces,repr(row)
        maps={name:t.as_mapping() if hasattr(t,'as_mapping') else dict(t) for name,t in traces.items()}
        raw.append({'lambda':lam,'seed':seed,'results':row.results,'traces':maps})
        for name,t in maps.items():
            q=t['queries'];e=np.asarray(t['errors'] if cfg['problem_type']=='pe' else t['policy_value_errors'])
            for tau in ([.01,.001,.0001] if args.run_id=='05_threshold' else [.001]):
                summary.append({'lambda':lam,'seed':seed,'algorithm':name,'threshold':tau,'stable_hit':stable(q,e,tau),'first_hit':first(q,e,tau),'final_error':float(e[-1]),'max_error':float(np.nanmax(e)),'last_query':int(q[-1])})
    save(out/'raw-traces.json',raw);save(out/'summary.json',summary)
    if inner_diagnostics:save(out/'inner-diagnostics.json',inner_diagnostics)
    if outer_counts:
        save(out/'outer-diagnostics.json',{'per_row_counts':{k:v.tolist() for k,v in outer_counts.items()},'initial_true_products_per_row':1 if cfg['problem_type']=='pe' else 2,'events':outer_diagnostics,'axis_definition':'Reported active-attempt index: excludes initialization, includes rejected attempts. Batched candidate true products execute on all lanes, including inactive lanes; restoration reuses saved operator data and adds zero products. Reference construction and diagnostic policy evaluation excluded.'})
    sources={str(p.relative_to(SOURCE)).replace('\\','/'):sha(p) for p in (SOURCE/'besp').rglob('*.py')}
    info={'run_id':args.run_id,'elapsed_seconds':time.time()-start,'command':sys.argv,'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ['jax','jaxlib','numpy','scipy']},'jax_enable_x64':bool(jax.config.jax_enable_x64),'devices':[str(d) for d in jax.devices()],'source_hashes':sources,'config_sha256':sha(cfg_path),'harness_sha256':sha(__file__),'inner_loop_parity':parity,'inner_loop':'JIT-compiled equivalent inner while loop; outer dispatcher with read-only diagnostics','outer_instrumentation':instrumentation,'reference_identity_claimed':False,'all_requested_rows_present':len(raw)==len(cfg['lambdas']),'float_precision':'float64','target_scope':'Residual-certified target reconstruction'}
    info['executed_harness_sha256']=EXECUTED_HARNESS_SHA256
    if inner_diagnostics:info['inner_cap_hit_count']=sum(sum(x['cap_hits']) for x in inner_diagnostics)
    save(out/'solver-settings.json',info)
    print('DONE',args.run_id,'seconds',info['elapsed_seconds'],'rows',len(raw),flush=True)

if __name__=='__main__':main()

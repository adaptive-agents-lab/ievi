"""Check that execution-only accelerations preserve the solver outputs."""
import argparse,json
from pathlib import Path
import run_practical as runner
import numpy as np
import jax.numpy as jnp

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()

rng=np.random.default_rng(413)
p=rng.uniform(size=(3,4,4));p/=p.sum(axis=-1,keepdims=True)
ph=rng.uniform(size=(3,4,4));ph/=ph.sum(axis=-1,keepdims=True)
r=rng.normal(size=(3,4));g=np.array([.9,.95,.99])
v=np.linalg.solve(np.eye(4)[None]-g[:,None,None]*p,r[...,None])[...,0]
inputs=[jnp.array(x) for x in [p,ph,r,g,v]]
kw=dict(max_queries=40,value_norm='L1',value_normalized=True,h_init=jnp.array([1e6]*3),h_min=jnp.ones(3),stall_patience=jnp.array([1,2,3]),decay_factor=jnp.array([.1,.2,.03]))
q0,e0=runner.dd._batched_waterfall_pe_common(*inputs,**kw)
transform=runner.instrument_outer('pe')
q1,e1=runner.dd._batched_waterfall_pe_common(*inputs,**kw)
assert np.array_equal(q0,q1)
assert np.array_equal(e0,e1,equal_nan=True)
report={'pe_trace_collector_bitwise_equal':True,'pe_fixture':'3 rows, 4 states, heterogeneous discounts, patience and decay,40 steps','pe_transformation':transform,'control_inner_loop':runner.check_inner_parity(),'executed_harness_sha256':runner.EXECUTED_HARNESS_SHA256}
out=args.output
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))

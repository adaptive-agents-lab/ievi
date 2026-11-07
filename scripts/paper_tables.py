"""Aggregate scientific data into eleven empirical tables."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import t



def rounded_values(spec, values):
    """Round aggregate values using explicit numeric display specifications."""
    if values == [None]:
        return [None]
    if spec["kind"] != "number" or len(values) != len(spec["precision"]):
        raise ValueError("Cell cardinality does not match its numeric specification")
    exponent = spec.get("exponent", 0) if spec["layout"] == "scientific" else 0
    return [float(f"{value / 10**exponent:.{precision}f}e{exponent}")
            for value, precision in zip(values, spec["precision"])]


def format_cell(spec, values):
    """Generate a mathematical cell from data and a display specification."""
    if spec["kind"] == "text":
        return spec["text"]
    if values == [None]:
        return spec.get("display", "---")
    rounded = rounded_values(spec, values)
    exponent = spec.get("exponent", 0) if spec["layout"] == "scientific" else 0
    tokens = []
    for value, precision in zip(rounded, spec["precision"]):
        rendered = f"{value / 10**exponent:.{precision}f}"
        if float(rendered) == 0:
            rendered = rendered.removeprefix("-")
        if spec.get("group_thousands") and precision == 0:
            rendered = f"{int(round(value)):,}".replace(",", r"{,}")
        tokens.append(rendered)
    layout = spec["layout"]
    if layout == "scalar":
        body = tokens[0]
    elif layout == "scientific":
        body = tokens[0] + r"\times10^{" + str(exponent) + "}"
    elif layout == "mean_se":
        body = tokens[0] + r" \pm " + tokens[1]
    elif layout == "estimate_interval":
        body = tokens[0] + " [" + tokens[1] + r",\," + tokens[2] + "]"
    elif layout == "interval":
        body = "[" + tokens[0] + r",\," + tokens[1] + "]"
    else:
        raise ValueError(f"Unsupported cell layout: {layout}")
    return r"\(" + body + r"\)"


def reference_values(spec):
    return spec["values"] if spec["kind"] == "number" else [None]


def generate(dataset,output):
    ROOT = Path(__file__).resolve().parents[1]
    OUT = output
    OUT.mkdir(parents=True,exist_ok=False)
    COVERAGE = json.loads((ROOT / "docs/coverage/coverage.json").read_text())
    TABLES = {x["paper_label"]: x for x in COVERAGE["table_inventory"] if x["classification"] == "empirical"}
    INPUTS = []

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def recomputed(task, relative, optional=False):
        from scripts.reference_dataset import resolve_attempt
        path, record = resolve_attempt(dataset,task)
        source=path.parent/relative
        actual=sha(source)
        if record['outputs'][relative]['sha256']!=actual:
            raise ValueError(f'Accepted file hash mismatch: {source}')
        INPUTS.append({'task':task,'path':str(source),'sha256':actual,
                       'execution_record':str(path),'execution_sha256':sha(path),
                       'identity_sha256':record['identity_sha256']})
        return pd.read_csv(source)

    control = recomputed("planning-control","control_accuracy/raw/control_accuracy_summary.csv")
    horizon = recomputed("planning-horizon","horizon_scaling/data/raw/records.csv")
    slopes = recomputed("planning-horizon","horizon_scaling/data/derived/slope_summary.csv")
    restoration = recomputed("restoration-common","common/control_accuracy_ievi_summary.csv")
    timing = recomputed("planning-timing","wallclock/data/raw/benchmark_runs.csv",optional=True)
    assert len(control)==95 and len(horizon)==1680 and len(restoration)==38
    assert restoration.data_source.eq("incumbent_restoration").all()

    def blocks(table):
        return [block["rows"] for block in table["blocks"]]

    CELLS=[]
    def compare(label, block, row, column, expected, actual, name, kind="rounded", pending=False):
        expected_values = reference_values(expected)
        values = list(actual) if isinstance(actual,(list,tuple,np.ndarray)) else [actual]
        values = [None if x is None or (isinstance(x,(float,np.floating)) and math.isnan(x)) else float(x) for x in values]
        nohit=expected["kind"] == "missing"
        tolerances=[]
        if pending:
            status="pending_recomputed_timing"; matches=None
        elif nohit:
            matches=values==[None]; status="match" if matches else "difference"
        else:
            exponent = expected.get("exponent", 0) if expected["layout"] == "scientific" else 0
            for value, precision in zip(expected_values, expected["precision"]):
                unit = 10.0**(exponent - precision)
                tolerances.append(0.0 if kind == "exact" else (1e-12 if value == 0 else unit / 2))
            matches=len(values)==len(expected_values) and all(a is not None and abs(a-e)<=tol+1e-12*max(abs(e),1) for a,e,tol in zip(values,expected_values,tolerances))
            status="match" if matches else ("new_timing_measurement" if kind=="timing" else "difference")
        CELLS.append({"table":label,"block":block,"row":row,"column":column,"metric":name,
                      "paper_cell":format_cell(expected, expected_values),"paper_values":expected_values,
                      "recomputed_values":values if not pending else None,"absolute_tolerances":tolerances,
                      "comparison_kind":kind,"matches_printed_value":matches,"status":status})

    def pick(df,lam,alg,tolerance):
        q=df[(df["lambda"]==lam)&(df.algorithm==alg)&(df.delta_solve==tolerance)]
        assert len(q)==1
        return q.iloc[0]

    for label in ("tab:final-control-counts","tab:final-control-costs","tab:final-tolerance-ievi","tab:final-tolerance-osvi","tab:incumbent-restoration-comparison"):
        for ri,row in enumerate(blocks(TABLES[label])[0],1):
            lam=row[0]["values"][0]; a=pick(control,lam,"IEVI_Waterfall_common_VI",.01)
            at=pick(control,lam,"IEVI_Waterfall_common_VI",1e-8)
            o=pick(control,lam,"OSVI_common_VI",.01); ot=pick(control,lam,"OSVI_common_VI",1e-8)
            if label.endswith("control-counts"):
                vals=[o.stable_N_P,o.stable_N_Phat,a.stable_N_P,a.stable_N_Phat]; kinds=["exact"]*4
                names=["OSVI true calls","OSVI surrogate calls","adaptive true calls","adaptive surrogate calls"]
            elif label.endswith("control-costs"):
                k=(49-a.stable_N_P)/a.stable_N_Phat if a.stable_N_P<49 else None
                vals=[k,(a.stable_N_P+.001*a.stable_N_Phat)/49,(a.stable_N_P+.01*a.stable_N_Phat)/49]
                kinds=["rounded"]*3; names=["kappa_star","cost ratio .001","cost ratio .01"]
            elif label.endswith("tolerance-ievi"):
                assert a.stable_N_P==at.stable_N_P
                vals=[a.stable_N_P,at.stable_N_Phat,a.stable_N_Phat,at.stable_N_Phat/a.stable_N_Phat]
                kinds=["exact"]*3+["rounded"]; names=["shared true calls","tight surrogate calls","loose surrogate calls","work ratio"]
            elif label.endswith("tolerance-osvi"):
                assert o.stable_N_P==ot.stable_N_P or (pd.isna(o.stable_N_P) and pd.isna(ot.stable_N_P))
                vals=[o.stable_N_P,ot.stable_N_Phat,o.stable_N_Phat]; kinds=["exact"]*3; names=["shared true calls","tight surrogate calls","loose surrogate calls"]
            else:
                b=pick(restoration,lam,"IEVI_Waterfall_common_VI",.01); bt=pick(restoration,lam,"IEVI_Waterfall_common_VI",1e-8)
                assert b.stable_N_P==bt.stable_N_P
                vals=[a.stable_N_P,at.stable_N_Phat,a.stable_N_Phat,b.stable_N_P,bt.stable_N_Phat,b.stable_N_Phat]
                kinds=["exact"]*6; names=["primary true","primary tight surrogate","primary loose surrogate","restoration true","restoration tight surrogate","restoration loose surrogate"]
            compare(label,0,ri,0,row[0],lam,"lambda","exact")
            for ci,(expected,value,kind,name) in enumerate(zip(row[1:],vals,kinds,names),1):
                compare(label,0,ri,ci,expected,value,name,kind)

    for label,schedule in [("tab:final-horizon-fixed","fixed80"),("tab:final-horizon-growing","linear16")]:
        for bi,block in enumerate(blocks(TABLES[label])):
            for ri,row in enumerate(block,1):
                H=row[0]["values"][0]; compare(label,bi,ri,0,row[0],H,"H","exact")
                algorithms=["VI","OS-VI"] if bi==0 else ["IE-VI Eq16","IE-VI Joint"]
                for ci,algorithm in enumerate(algorithms,1):
                    q=horizon[(horizon.schedule==schedule)&(horizon.H==H)&(horizon.algorithm==algorithm)]
                    assert len(q)==30 and q.replication.nunique()==30
                    compare(label,bi,ri,ci,row[ci],np.quantile(q.N_P,[.5,.25,.75]),algorithm+" median,q25,q75","exact")
                if bi==0:
                    compare(label,bi,ri,3,row[3],q.E_infinity.median(),"median model error")

    label="tab:final-horizon-slopes"
    quantities=["E_infinity","N_P::VI","N_P::OS-VI","N_P::IE-VI Eq16","N_P::IE-VI Joint"]
    SLOPE_RECOMPUTATION=[]
    for ri,row in enumerate(blocks(TABLES[label])[0],1):
        schedule="fixed80" if ri<=5 else "linear16"; quantity=quantities[(ri-1)%5]
        algorithm="IE-VI Joint" if quantity=="E_infinity" else quantity.split("::")[1]
        field="E_infinity" if quantity=="E_infinity" else "N_P"
        matrix=horizon[(horizon.schedule==schedule)&(horizon.algorithm==algorithm)].pivot(index="replication",columns="H",values=field)
        assert matrix.shape==(30,7)
        x=np.log(matrix.columns.to_numpy(float)); centered=x-x.mean(); denom=np.dot(centered,centered)
        point=float(np.dot(np.log(np.median(matrix.to_numpy(),axis=0)),centered)/denom)
        rng=np.random.default_rng(4400 if quantity=="E_infinity" else 4500)
        indices=rng.integers(0,30,(4000,30))
        medians=np.median(matrix.to_numpy()[indices],axis=1)
        fits=np.log(medians)@centered/denom
        low,high=np.quantile(fits,[.025,.975])
        recomputed_saved=slopes[(slopes.schedule==schedule)&(slopes.quantity==quantity)&(slopes.grid=="all_H")].iloc[0]
        SLOPE_RECOMPUTATION.append({"schedule":schedule,"quantity":quantity,"resamples":4000,
            "matches_recomputed_saved":bool(np.allclose([point,low,high],[recomputed_saved.slope,recomputed_saved.ci_low,recomputed_saved.ci_high],rtol=0,atol=1e-12))})
        compare(label,0,ri,2,row[2],point,quantity+" slope")
        compare(label,0,ri,3,row[3],[low,high],quantity+" paired bootstrap95CI")

    methods=["VI","IEVI_VI_Ks4","IEVI_VI_Ks8","IEVI_VI_Ks16","IEVI_VI_Ks32"]
    if timing is not None:
        assert len(timing)==50 and set(timing.method)==set(methods) and timing.converged.all()
        recomputed_setup=timing.solver_seconds+timing.true_setup_seconds+timing.surrogate_setup_seconds
        assert np.allclose(timing.setup_plus_solver_seconds,recomputed_setup,rtol=0,atol=1e-10)
        timing=timing.copy()
        timing["setup_plus_solver_seconds"]=recomputed_setup
    for label,field in [("tab:final-runtime","solver_seconds"),("tab:final-runtime-with-setup","setup_plus_solver_seconds"),("tab:final-runtime-counts",None)]:
        for ri,row in enumerate(blocks(TABLES[label])[0],1):
            method=methods[ri-1]
            q=None if timing is None else timing[timing.method==method].sort_values("repetition")
            if q is not None:
                assert len(q)==10 and list(q.repetition)==list(range(10))
            if field:
                stats=[None,None] if q is None else [q[field].mean(),q[field].std(ddof=1)/np.sqrt(10)]
                compare(label,0,ri,1,row[1],stats,method+" mean,SE","timing",q is None)
                if method=="VI":
                    compare(label,0,ri,2,row[2],None,"VI speedup undefined","exact")
                else:
                    if q is None:
                        ratios=[None]*3
                    else:
                        v=timing[timing.method=="VI"].sort_values("repetition")[field].to_numpy()/q[field].to_numpy()
                        mean=v.mean(); margin=t.ppf(.975,9)*v.std(ddof=1)/np.sqrt(10); ratios=[mean,mean-margin,mean+margin]
                    compare(label,0,ri,2,row[2],ratios,method+" paired ratio mean,CI","timing",q is None)
            else:
                for ci,column in [(1,"true_operator_calls"),(2,"surrogate_operator_calls"),(3,"final_relative_true_residual")]:
                    value=None if q is None else q[column].max()
                    if q is not None and ci<3: assert q[column].nunique()==1
                    compare(label,0,ri,ci,row[ci],value,method+" "+column,"exact" if ci<3 else "rounded",q is None)

    counts={status:sum(c["status"]==status for c in CELLS) for status in sorted({c["status"] for c in CELLS})}
    report={"schema_version":1,"checked_at":datetime.now(timezone.utc).isoformat(),
        "paper_source":"paper/manuscript","script_sha256":sha(Path(__file__)),
        "table_source_snapshot":str((ROOT/"docs/coverage/coverage.json").relative_to(ROOT)),"table_source_snapshot_sha256":sha(ROOT/"docs/coverage/coverage.json"),
        "scope":"All 11 empirical tables; every numerical and no-hit data cell, including numeric row keys, ratios, quartiles and paired confidence intervals. Categorical labels, headers and captions provide context and are not counted as numerical cells. Pure aggregation; no solver runs.",
        "recomputed_timing_available":timing is not None,"counts":counts,
        "status":"differences" if counts.get("difference",0) else ("pending_timing" if timing is None else "completed"),
        "completed_scientific_cells_match":counts.get("difference",0)==0,
        "all_required_scientific_cells_checked":timing is not None,
        "scientific_discrete_and_rounded_values_pass":None if timing is None else counts.get("difference",0)==0,
        "recomputed_sources":INPUTS,"slope_independent_recomputation":SLOPE_RECOMPUTATION,
        "tables":[{"label":label,"cell_count":sum(c["table"]==label for c in CELLS),
                   "counts":{s:sum(c["table"]==label and c["status"]==s for c in CELLS) for s in counts}} for label,tab in TABLES.items()],
        "cells":CELLS,
        "interpretation":"Exact discrete scientific values and rounded table statistics are compared to the paper. Recomputed elapsed times and paired runtime speedups are new machine-specific measurements; differences are reported without treating reference seconds as a pass band. Timing count/residual cells still require exact/printed-precision agreement. Data files are bound to their scientific record hashes."}
    # Write mathematical table rows from structured labels and numeric aggregates.
    TABLE_EXPORTS=[]
    for label,table in TABLES.items():
        exported=[]
        for bi,block in enumerate(blocks(table)):
            rendered=[]
            for ri,row in enumerate(block,1):
                new=[format_cell(spec, reference_values(spec)) for spec in row]
                for cell in (c for c in CELLS if c['table']==label and c['block']==bi and c['row']==ri):
                    spec=row[cell['column']]
                    new[cell['column']]=format_cell(spec,cell['recomputed_values'])
                    cell['generated_cell']=new[cell['column']]
                    cell['generated_values']=rounded_values(spec,cell['recomputed_values'])
                    tol=cell['absolute_tolerances']
                    cell['generated_matches_aggregation']=cell['recomputed_values']==cell['generated_values'] if cell['comparison_kind']=='exact' or cell['recomputed_values']==[None] else all(abs(a-b)<=t+1e-12*max(abs(a),1) for a,b,t in zip(cell['recomputed_values'],cell['generated_values'],tol))
                    if not cell['generated_matches_aggregation']:
                        raise ValueError('Generated exact cell changed: '+str(cell))
                rendered.append(' & '.join(x.strip() for x in new)+r' \\')
            text='\n'.join(rendered)+'\n'
            exported.append(text)
            if len(blocks(table))>1:
                (OUT/(label.removeprefix('tab:')+f'-block-{bi}.tex')).write_text(text,encoding='utf-8')
        target=OUT/(label.removeprefix('tab:')+'.tex')
        target.write_text('\n'.join('% block '+str(i)+'\n'+x for i,x in enumerate(exported)),encoding='utf-8')
        TABLE_EXPORTS.append({'label':label,'path':target.name,'sha256':sha(target),'blocks':len(exported),
                              'cell_count':sum(c['table']==label for c in CELLS)})
    report['generated_tables']=TABLE_EXPORTS
    report['generated_cells']=len(CELLS)
    report['all_generated_cells_match_aggregation']=all(c['generated_matches_aggregation'] for c in CELLS)
    assert len(CELLS)==628 and len(TABLE_EXPORTS)==11
    (OUT/"table-reconciliation.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    lines=["# Paper table reconciliation","",f"{len(CELLS)} data cells across all 11 empirical tables.","",f"Status: **{report['status']}**. Counts: "+", ".join(f"{k}: {v}" for k,v in counts.items())+".","","| Table | Cells | Result |","|---|---:|---|"]
    for table in report["tables"]:
        lines.append("| "+table["label"]+" | "+str(table["cell_count"])+" | "+", ".join(f"{k}: {v}" for k,v in table["counts"].items() if v)+" |")
    lines += ["","All numerical inputs match verified scientific records. The method comparison includes primary and incumbent restoration outcomes. Horizon quantiles and all ten full-range slopes/4,000-resample intervals were independently aggregated from recomputed records.","","Recomputed timing statistics are separate measurements, not deterministic equality targets. "+("Zero cells remain pending. Setup-plus-solver sums, means, standard errors and paired speedup intervals were recomputed from all 50 recomputed measurements. " if timing is not None else "Only timing-dependent cells remain pending while that run is incomplete. ")+"The JSON records every expected/recomputed cell, tolerance, source hash, and difference."]
    differences=[c for c in CELLS if c["status"]=="difference"]
    if differences:
        lines += ["","## Differences"]+[f"- {c['table']} block {c['block']}, row {c['row']}, column {c['column']}: paper {c['paper_values']}; recomputed {c['recomputed_values']}." for c in differences]
    (OUT/"table-reconciliation.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    primary=control.copy()
    restored=pd.concat([control[control.algorithm!='IEVI_Waterfall_common_VI'],restoration],ignore_index=True)
    assert len(restored)==95 and len(restored[restored.algorithm!='IEVI_Waterfall_common_VI'])==57
    primary.to_csv(OUT/'primary-common-summary.csv',index=False)
    restored.to_csv(OUT/'restoration-common-summary.csv',index=False)
    keys=['lambda','algorithm','delta_solve']
    primary.merge(restored,on=keys,suffixes=('_primary','_restoration'),validate='one_to_one').to_csv(OUT/'primary-restoration-comparison.csv',index=False)

    return report

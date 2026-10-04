import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
from dataclasses import asdict
import json,time,hashlib,copy
from pathlib import Path
import numpy as np
from .engine import Config,Ensemble
from .problems import make_problem

POLICIES=['adaptive','search_only','sample_always','fixed_25','fixed_50','fixed_75','random_50','uncorrected','adaptive_random','fixed_50_corrected','adaptive_total','adaptive_max','sample_random','sample_maximin','resample_only']

def decision(policy,diag,available,query_cost,iteration,rng):
    S=float(np.mean(diag['search'][available]));D=float(np.mean(diag['data'][available]))
    if policy in ['search_only','resample_only']:sample=False
    elif policy in ['sample_always','sample_random','sample_maximin']:sample=True
    elif policy.startswith('fixed_'):
        rate=float(policy.split('_')[1])/100
        sample=int((iteration+1)*rate)>int(iteration*rate)
    elif policy=='random_50':sample=rng.random()<0.5
    else:
        if policy=='uncorrected':D=float(np.mean(diag['between'][available]))
        if policy=='adaptive_max':D=float(np.max(diag['data'][available]))
        sample=D/(1+query_cost)>S
    if policy in ['adaptive_random','sample_random']:q=int(rng.choice(available))
    else:
        scores=diag['total'] if policy in ['sample_always','fixed_25','fixed_50','fixed_75','random_50','adaptive_total'] else diag['data']
        q=int(available[np.argmax(scores[available])])
    return sample,q,S,D

def run_one(spec,outdir=None,return_state=False):
    cfg=Config(**spec.get('engine',{}))
    data=make_problem(spec['problem'],spec['seed'],spec.get('noise',0.05),spec.get('pool_size',384))
    X=data['X'];y=data['y'];seed=spec['seed'];policy=spec['policy'];cost=spec['query_cost'];budget=spec.get('budget',48.)
    rng=np.random.default_rng(seed+73031);init_rng=np.random.default_rng(seed+719)
    ids=np.array(init_rng.choice(len(X),spec.get('initial_labels',16),replace=False),dtype=np.int64)
    acquired=np.zeros(len(X),bool);acquired[ids]=True
    start=time.perf_counter();engine=Ensemble(X,seed+3901,cfg)
    trace=[];spent=0.;iteration=0;checkpoint=0
    # Initialization and fixed warm-up are common to all policies and reported separately.
    for _ in range(spec.get('warmup',2)):engine.step(ids,y)
    init_seconds=time.perf_counter()-start
    def record(action,S=0.,D=0.):
        model=engine.select(ids,y)
        pred=engine.predict(model,data['test_X']);opred=engine.predict(model,data['ood_X'])
        nmse=float(np.mean((pred-data['test_y'])**2)/(np.var(data['test_y'])+1e-12))
        ood=float(np.mean((opred-data['ood_y'])**2)/(np.var(data['ood_y'])+1e-12))
        trace.append({'cost':spent,'labels':len(ids),'blocks':engine.blocks,'nmse':nmse,'ood_nmse':ood,'action':action,'search_signal':S,'data_signal':D,'work':engine.work(),'seconds':time.perf_counter()-start})
    record('initial')
    while spent+1<=budget+1e-9:
        diag=engine.diagnostics();avail=np.flatnonzero(~acquired)
        sample,q,S,D=decision(policy,diag,avail,cost,iteration,rng)
        if policy=='sample_maximin':
            distances=np.sum((X[avail,None,:]-X[ids][None,:,:])**2,axis=2)
            q=int(avail[np.argmax(np.min(distances,axis=1))])
        if sample and spent+1+cost<=budget+1e-9:
            ids=np.r_[ids,q];acquired[q]=True;spent+=cost;action='sample'
        else:action='search'
        if policy=='resample_only':engine.seed+=1013
        engine.step(ids,y);spent+=1;iteration+=1
        if spent>=checkpoint+budget/8 or spent+1>budget or return_state:
            record(action,S,D);checkpoint=spent
    model=engine.select(ids,y)
    final=dict(trace[-1]);result={'spec':spec,'config':asdict(cfg),'initial_seconds':init_seconds,'elapsed_seconds':time.perf_counter()-start,'initial_cost':spec.get('initial_labels',16)*cost+spec.get('warmup',2)+1/cfg.generations,'final':final,'trace':trace,'acquired_indices':ids.tolist(),'model':{'ops':model[1].tolist(),'constants':model[2].tolist(),'intercept':model[3],'scale':model[4]}}
    if outdir:
        key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
        path=Path(outdir)/f'{key}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,separators=(',',':')));tmp.replace(path)
    if return_state:return result,engine,data,ids
    return result

#!/usr/bin/env python3
"""Equal-cost counterfactual branches; audit targets never inform live actions."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse,json,copy,time,concurrent.futures,hashlib
from pathlib import Path
import numpy as np
from sos.engine import Ensemble,Config
from sos.problems import make_problem,FUNCTIONS
from sos.experiment import decision

def audit(spec,out):
 key=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16];path=Path(out)/(key+'.json')
 if path.exists():return key
 d=make_problem(spec['problem'],spec['seed'],spec['noise']);X,y=d['X'],d['y'];seed=spec['seed']
 ids=np.array(np.random.default_rng(seed+719).choice(len(X),16,replace=False));acq=np.zeros(len(X),bool);acq[ids]=True
 c=Config(population=32,generations=2,depth=4,bootstraps=3,restarts=3);e=Ensemble(X,seed+3901,c)
 for _ in range(2):e.step(ids,y)
 rng=np.random.default_rng(seed+73031);spent=0;iteration=0;records=[];checkpoints=[0,16,48]
 def risk(ee,ii):
  m=ee.select(ii,y);p=ee.predict(m,d['test_X']);return float(np.mean((p-d['test_y'])**2)/(np.var(d['test_y'])+1e-12))
 while checkpoints:
  if spent>=checkpoints[0]:
   checkpoints.pop(0);diag=e.diagnostics();avail=np.flatnonzero(~acq)
   sample,q,S,D=decision('adaptive',diag,avail,1.,iteration,rng);values=[]
   for rep in range(4):
    search=copy.deepcopy(e)
    for b,g in enumerate(search.islands):
     for r,island in enumerate(g):island.rng=np.random.default_rng(972331+seed*10000+iteration*100+rep*10+b*3+r)
    queried=copy.deepcopy(search);refreshed=copy.deepcopy(search);refreshed.seed+=1013;newids=np.r_[ids,q]
    for _ in range(5):search.step(ids,y)
    for _ in range(4):queried.step(newids,y)
    for _ in range(5):refreshed.step(ids,y)
    values.append([risk(search,ids),risk(queried,newids),risk(refreshed,ids)])
   records.append({'spent':spent,'labels':len(ids),'current_nmse':risk(e,ids),'search_signal':S,'data_signal':D,'between_signal':float(diag['between'][avail].mean()),'chosen_sample':bool(sample),'query_index':q,'branch_cost':5.,'branch_nmse':values})
  if not checkpoints:break
  diag=e.diagnostics();sample,q,_,_=decision('adaptive',diag,np.flatnonzero(~acq),1.,iteration,rng)
  if sample:ids=np.r_[ids,q];acq[q]=True;spent+=1
  e.step(ids,y);spent+=1;iteration+=1
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps({'spec':spec,'states':records},separators=(',',':')));tmp.replace(path);return key

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--out',default='results/audit');a=p.parse_args();Path(a.out).mkdir(exist_ok=True,parents=True)
 lock_handle=(Path(a.out)/'.audit.lock').open('a+')
 if os.name=='posix':
  import fcntl
  fcntl.flock(lock_handle.fileno(),fcntl.LOCK_EX)
 probs=['poly3','poly4','sinmix','sincos','sin2','interaction','rational','poly2d','trig3d','rational3d','oscillator','drag']
 specs=[{'problem':pr,'seed':s,'noise':n} for pr in probs for s in range(700,710) for n in [0.,.1]]
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
  futures=[ex.submit(audit,s,a.out) for s in specs]
  for i,f in enumerate(concurrent.futures.as_completed(futures)):f.result();print(i+1,len(specs),flush=True)
if __name__=='__main__':main()

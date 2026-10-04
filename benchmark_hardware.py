#!/usr/bin/env python3
"""Compare a batched Metal tree evaluator with the compiled CPU evaluator."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,time,platform,argparse
from pathlib import Path
import numpy as np
from sos.engine import Island,Config,evaluate

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',default='results/hardware.json');a=p.parse_args()
 import torch
 torch.set_num_threads(1);records=[]
 for population,n in [(32,384),(288,384),(288,4096)]:
  X=np.random.default_rng(3).uniform(-1,1,(n,3));island=Island(X,42,Config(population=population,depth=4));o,c=island.ops,island.constants
  def cpu():return evaluate(o,c,X)
  cpu();t=[]
  for _ in range(10):start=time.perf_counter();cpu();t.append(time.perf_counter()-start)
  rec={'trees':population,'points':n,'cpu_seconds_median':float(np.median(t)),'cpu_precision':'float64'}
  if torch.backends.mps.is_available():
   dev=torch.device('mps');O=torch.tensor(o,dtype=torch.int64,device=dev);C=torch.tensor(c,dtype=torch.float32,device=dev);T=torch.tensor(X,dtype=torch.float32,device=dev)
   def metal():
    values=[None]*o.shape[1]
    for j in range(o.shape[1]-1,-1,-1):
     op=O[:,j,None];z=C[:,j,None].expand(population,n)
     for k in range(X.shape[1]):z=torch.where(op==k+1,T[:,k][None,:],z)
     if 2*j+1<o.shape[1]:
      l=values[2*j+1];r=values[2*j+2];den=torch.where(r.abs()>1e-6,r,torch.ones_like(r))
      for tag,v in [(10,l+r),(11,l-r),(12,l*r),(13,l/den),(14,l.sin()),(15,l.cos())]:z=torch.where(op==tag,v,z)
     values[j]=z.clamp(-1e6,1e6)
    return values[0]
   metal();torch.mps.synchronize();tm=[]
   for _ in range(10):
    start=time.perf_counter();q=metal();torch.mps.synchronize();tm.append(time.perf_counter()-start)
   diff=np.abs(q.cpu().numpy()-cpu());rec.update({'mps_seconds_median':float(np.median(tm)),'mps_precision':'float32','absolute_error_median':float(np.median(diff)),'absolute_error_max':float(diff.max()),'cpu_over_mps_time':float(np.median(t)/np.median(tm))})
  records.append(rec)
 result={'platform':platform.platform(),'processor':platform.processor(),'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'mps_available':torch.backends.mps.is_available(),'records':records,'scope':'Comparison of these two implementations; GPU evaluator uses masked batched operators. This does not establish a hardware-wide speed ranking.'}
 out=Path(a.out);out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()

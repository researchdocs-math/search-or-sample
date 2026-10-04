#!/usr/bin/env python3
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
 os.environ[name]='1'
import argparse,json,hashlib,time,concurrent.futures,platform
from pathlib import Path
from sos.experiment import run_one

def worker(args):
 spec,out=args
 t=time.perf_counter();r=run_one(spec,out)
 return spec['problem'],spec['policy'],spec['seed'],r['final']['nmse'],round(time.perf_counter()-t,2)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--out',required=True);a=p.parse_args()
 cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 # Serialize identical configurations when independent launchers share an output tree.
 lock_handle=(out/('.run-'+hashlib.sha256(Path(a.config).read_bytes()).hexdigest()[:16]+'.lock')).open('a+')
 if os.name=='posix':
  import fcntl
  fcntl.flock(lock_handle.fileno(),fcntl.LOCK_EX)
 specs=[]
 for problem in cfg['problems']:
  for seed in cfg['seeds']:
   for noise in cfg['noise_levels']:
    for cost in cfg['query_costs']:
     for policy in cfg['policies']:
      s={**cfg.get('common',{}),'problem':problem,'seed':seed,'noise':noise,'query_cost':cost,'policy':policy}
      key=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()[:16]
      if not (out/f'{key}.json').exists():specs.append(s)
 print(json.dumps({'remaining':len(specs),'workers':a.workers,'platform':platform.platform()}),flush=True)
 start=time.perf_counter()
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as ex:
  for i,r in enumerate(ex.map(worker,[(s,str(out)) for s in specs])):
   print(json.dumps({'done':i+1,'total':len(specs),'result':r,'elapsed':round(time.perf_counter()-start,1)}),flush=True)
if __name__=='__main__':main()

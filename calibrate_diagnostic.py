#!/usr/bin/env python3
"""Monte Carlo verification of the nested random-effects calculation."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import f, beta

def main():
 p=argparse.ArgumentParser();p.add_argument('--replicates',type=int,default=250000);p.add_argument('--out',default='results/calibration.json');a=p.parse_args()
 rng=np.random.default_rng(773311);records=[]
 for B,R in [(2,2),(3,3),(3,5),(5,3),(8,5)]:
  for tau2 in [0.,.1,1.,10.]:
   z=np.sqrt(tau2)*rng.normal(size=(a.replicates,B,1))+rng.normal(size=(a.replicates,B,R))
   W=z.var(axis=2,ddof=1).mean(axis=1);C=z.mean(axis=2).var(axis=1,ddof=1);S=W/R;D=C-S
   nu1,nu2=B-1,B*(R-1)
   threshold=nu1/(nu1+nu2)
   null_bias=(nu1+nu2)/(R*nu2)*(beta.sf(threshold,nu1/2+1,nu2/2)-beta.sf(threshold,nu1/2,nu2/2))
   for cost in [.25,1.,4.]:
    sample=np.maximum(D,0)/(1+cost)>S
    records.append({'B':B,'R':R,'tau2':tau2,'sigma2':1.,'query_cost':cost,'replicates':a.replicates,'mean_W':float(W.mean()),'mean_C':float(C.mean()),'mean_raw_D':float(D.mean()),'mean_positive_D':float(np.maximum(D,0).mean()),'fraction_negative':float((D<0).mean()),'sample_rate':float(sample.mean()),'null_exact_positive_mean':float(null_bias) if tau2==0 else None,'null_exact_rate':float(f.sf(2+cost,B-1,B*(R-1))) if tau2==0 else None})
 out=Path(a.out);out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(records,indent=2));print('Saved',len(records),'calibration settings')
if __name__=='__main__':main()

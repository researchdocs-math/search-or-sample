"""Retrospective pool-based acquisition on two public engineering datasets."""
from pathlib import Path
import numpy as np

def make_engineering_problem(name,seed,noise=0.,pool_size=384,test_size=1536):
    if noise!=0:raise ValueError('Engineering data use their observed responses without added noise.')
    a=np.loadtxt(Path(__file__).resolve().parents[1]/'data'/f'{name}.csv',delimiter=',')
    rng=np.random.default_rng(830000+seed);order=rng.permutation(len(a))
    ntest=int(np.ceil(.25*len(a)));test=order[:ntest];pool=order[ntest:ntest+pool_size]
    X=a[pool,:-1];T=a[test,:-1];lo=X.min(0);hi=X.max(0)
    X=2*(X-lo)/np.maximum(hi-lo,1e-12)-1;T=2*(T-lo)/np.maximum(hi-lo,1e-12)-1
    return {'X':X,'y':a[pool,-1],'latent':a[pool,-1],'test_X':T,'test_y':a[test,-1],'ood_X':T,'ood_y':a[test,-1],'noise_sd':0.,'name':name,'pool_rows':pool,'test_rows':test}

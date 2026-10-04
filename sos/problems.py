"""Controlled equation discovery tasks and physically motivated response functions."""
import numpy as np

FUNCTIONS={
 'poly3':(1,lambda x:x[:,0]**3+x[:,0]**2+x[:,0]),
 'poly4':(1,lambda x:x[:,0]**4+x[:,0]**3+x[:,0]**2+x[:,0]),
 'sinmix':(1,lambda x:np.sin(x[:,0]**2)*np.cos(x[:,0])-1),
 'sincos':(1,lambda x:np.sin(x[:,0])+np.sin(x[:,0]**2+x[:,0])),
 'sin2':(2,lambda x:np.sin(x[:,0])+np.sin(x[:,1]**2)),
 'interaction':(2,lambda x:2*np.sin(x[:,0])*np.cos(x[:,1])),
 'rational':(2,lambda x:(x[:,0]+x[:,1])/(1+x[:,0]**2)),
 'poly2d':(2,lambda x:x[:,0]**2+x[:,0]*x[:,1]+x[:,1]),
 'trig3d':(3,lambda x:np.sin(x[:,0])+x[:,1]*x[:,2]),
 'rational3d':(3,lambda x:(x[:,0]*x[:,1])/(2+x[:,2]**2)),
 'oscillator':(2,lambda x:-0.4*x[:,1]-1.5*x[:,0]-0.7*x[:,0]**3),
 'drag':(2,lambda x:0.5*(x[:,0]+1.5)*(x[:,1]+1.5)**2),
 'pilot_cubic':(1,lambda x:0.8*x[:,0]**3-0.5*x[:,0]),
 'pilot_trig':(2,lambda x:np.sin(1.3*x[:,0])+0.7*x[:,1]),
 'pilot_ratio':(2,lambda x:x[:,0]/(2+x[:,1]**2)),
}

def make_problem(name,seed,noise=0.05,pool_size=384,test_size=1536):
    if name in ('airfoil','concrete'):
        from .engineering import make_engineering_problem
        return make_engineering_problem(name,seed,noise,pool_size,test_size)
    d,fun=FUNCTIONS[name]
    rng=np.random.default_rng(830000+seed)
    X=rng.uniform(-1,1,(pool_size,d));T=rng.uniform(-1,1,(test_size,d))
    O=rng.uniform(-1.5,1.5,(test_size,d))
    f=fun(X);truth=fun(T);oot=fun(O)
    scale=max(np.std(f),1e-8)
    y=f+rng.normal(0,noise*scale,len(X))
    return {'X':X,'y':y,'latent':f,'test_X':T,'test_y':truth,'ood_X':O,'ood_y':oot,'noise_sd':noise*scale,'name':name}

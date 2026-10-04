"""A reproducible, affine-scaled tree-GP engine with independent bootstrap islands."""
from dataclasses import dataclass
import numpy as np
from numba import njit

# Complete binary storage, with inactive descendants ignored semantically.
# terminals 0: constant, 1..d: variables; operators 10:+ 11:- 12:* 13:/
# 14:sin 15:cos 16:exp(clipped) 17:sqrt(abs).
@njit(cache=True)
def evaluate(ops, constants, X):
    ntrees, size = ops.shape
    n, d = X.shape
    out = np.empty((ntrees,n))
    v = np.empty(size)
    for k in range(ntrees):
        for i in range(n):
            for j in range(size-1,-1,-1):
                op=ops[k,j]
                if op==0: z=constants[k,j]
                elif op<10: z=X[i,op-1]
                elif op==10: z=v[2*j+1]+v[2*j+2]
                elif op==11: z=v[2*j+1]-v[2*j+2]
                elif op==12: z=v[2*j+1]*v[2*j+2]
                elif op==13:
                    den=v[2*j+2]
                    z=v[2*j+1]/den if abs(den)>1e-6 else v[2*j+1]
                elif op==14: z=np.sin(v[2*j+1])
                elif op==15: z=np.cos(v[2*j+1])
                elif op==16: z=np.exp(min(8.,max(-8.,v[2*j+1])))
                else: z=np.sqrt(abs(v[2*j+1]))
                v[j]=min(1e6,max(-1e6,z))
            out[k,i]=v[0]
    return out

@njit(cache=True)
def affine_score(pred, ids, y, complexity, penalty):
    k,n=pred.shape
    a=np.empty(k);b=np.empty(k);mse=np.empty(k);score=np.empty(k)
    ym=np.mean(y[ids]);var=np.mean((y[ids]-ym)**2)+1e-12
    for t in range(k):
        xm=0.
        for j in ids:xm+=pred[t,j]
        xm/=len(ids)
        xx=0.;xy=0.
        for j in ids:
            xx+=(pred[t,j]-xm)**2;xy+=(pred[t,j]-xm)*(y[j]-ym)
        b[t]=xy/(xx+1e-12);a[t]=ym-b[t]*xm
        err=0.
        for j in ids:err+=(a[t]+b[t]*pred[t,j]-y[j])**2
        mse[t]=err/len(ids)
        score[t]=mse[t]/var+penalty*complexity[t]
    return a,b,mse,score

@njit(cache=True)
def node_count(ops):
    out=np.zeros(len(ops),dtype=np.int32)
    for i in range(len(ops)):
        active=np.zeros(ops.shape[1],dtype=np.bool_);active[0]=True
        for j in range(ops.shape[1]):
            if active[j]:
                out[i]+=1
                if ops[i,j]>=10:
                    active[2*j+1]=True
                    if ops[i,j]<14:active[2*j+2]=True
    return out

@dataclass
class Config:
    population:int=48
    generations:int=3
    depth:int=4
    bootstraps:int=3
    restarts:int=3
    penalty:float=1e-4
    tournament:int=4
    mutation:float=0.75
    operators:tuple=(10,11,12,13,14,15)

class Island:
    def __init__(self,X,seed,config):
        self.X=X;self.rng=np.random.default_rng(seed);self.c=config
        self.size=2**(self.c.depth+1)-1
        self.ops=np.zeros((self.c.population,self.size),dtype=np.int16)
        self.constants=np.zeros_like(self.ops,dtype=float)
        for i in range(self.c.population):self.grow(self.ops[i],self.constants[i],0,0)
        self.pred=evaluate(self.ops,self.constants,X)
        self.work=len(self.ops)
        self.evaluated_points=len(self.ops)*len(X)
        self.a=self.b=self.mse=self.score=None
    def grow(self,o,c,j,depth,force=False):
        if depth>=self.c.depth or (not force and self.rng.random()<0.32):
            o[j]=self.rng.integers(0,self.X.shape[1]+1);c[j]=self.rng.uniform(-2,2)
        else:
            o[j]=self.rng.choice(self.c.operators)
            self.grow(o,c,2*j+1,depth+1)
            self.grow(o,c,2*j+2,depth+1)
    def active(self,o):
        ids=[0]
        for j in ids:
            if o[j]>=10:
                ids.append(2*j+1)
                if o[j]<14:ids.append(2*j+2)
        return ids
    def fit(self,ids,y):
        self.a,self.b,self.mse,self.score=affine_score(self.pred,ids,y,node_count(self.ops),self.c.penalty)
    def copy_sub(self,dsto,dstc,src_o,src_c,j,k):
        if j>=self.size:return
        # Source cannot extend below destination capacity.
        if 2*j+1>=self.size and src_o[k]>=10:
            dsto[j]=self.rng.integers(0,self.X.shape[1]+1);dstc[j]=self.rng.uniform(-2,2);return
        dsto[j]=src_o[k];dstc[j]=src_c[k]
        if src_o[k]>=10:
            self.copy_sub(dsto,dstc,src_o,src_c,2*j+1,2*k+1)
            if src_o[k]<14:self.copy_sub(dsto,dstc,src_o,src_c,2*j+2,2*k+2)
    def step(self,ids,y):
        self.fit(ids,y)
        for _ in range(self.c.generations):
            oo=np.empty_like(self.ops);cc=np.empty_like(self.constants)
            for t in range(self.c.population):
                sel=self.rng.integers(0,self.c.population,self.c.tournament)
                p=sel[np.argmin(self.score[sel])]
                oo[t]=self.ops[p];cc[t]=self.constants[p]
                j=self.rng.choice(self.active(oo[t]));dep=int(np.floor(np.log2(j+1)))
                if self.rng.random()<self.c.mutation:
                    self.grow(oo[t],cc[t],j,dep)
                else:
                    q=self.rng.integers(self.c.population);k=self.rng.choice(self.active(self.ops[q]))
                    self.copy_sub(oo[t],cc[t],self.ops[q],self.constants[q],j,k)
            pp=evaluate(oo,cc,self.X);self.work+=len(oo);self.evaluated_points+=len(oo)*len(self.X)
            aa,bb,mm,ss=affine_score(pp,ids,y,node_count(oo),self.c.penalty)
            # Elitist replacement preserves the best parent and keeps diverse genotypes.
            allscore=np.r_[self.score,ss];order=np.argsort(allscore,kind='stable')
            O=np.concatenate((self.ops,oo));C=np.concatenate((self.constants,cc));P=np.concatenate((self.pred,pp))
            A=np.r_[self.a,aa];B=np.r_[self.b,bb];M=np.r_[self.mse,mm]
            chosen=[];seen=set()
            for idx in order:
                key=O[idx].tobytes()+np.round(C[idx],5).tobytes()
                if key not in seen:seen.add(key);chosen.append(idx)
                if len(chosen)==self.c.population:break
            if len(chosen)<self.c.population:
                chosen += [int(x) for x in order[:self.c.population-len(chosen)]]
            ix=np.array(chosen)
            self.ops=O[ix];self.constants=C[ix];self.pred=P[ix]
            self.a=A[ix];self.b=B[ix];self.mse=M[ix];self.score=allscore[ix]
    def champion(self):
        i=int(np.argmin(self.score));return self.a[i]+self.b[i]*self.pred[i]
    def expression(self,i):
        def f(j):
            o=self.ops[i,j]
            if o==0:return f'{self.constants[i,j]:.6g}'
            if o<10:return f'x{o-1}'
            a=f(2*j+1)
            if o<14:return '('+a+{10:'+',11:'-',12:'*',13:'/'}[o]+f(2*j+2)+')'
            return {14:'sin',15:'cos',16:'exp_clip',17:'sqrt_abs'}[o]+'('+a+')'
        return f'{self.a[i]:.7g}+({self.b[i]:.7g})*'+f(0)

class Ensemble:
    def __init__(self,X,seed,config):
        self.X=X;self.seed=seed;self.c=config;self.blocks=0
        self.islands=[[Island(X,seed+1009*b+101*r,config) for r in range(config.restarts)] for b in range(config.bootstraps)]
    def bootstrap_ids(self,ids,b):
        # Fixed within an acquisition stage and identical across restart replicates.
        rng=np.random.default_rng(self.seed+100003*b+1013*len(ids))
        return rng.choice(ids,size=len(ids),replace=True)
    def fit(self,ids,y):
        for b,group in enumerate(self.islands):
            ii=self.bootstrap_ids(ids,b)
            for island in group:island.fit(ii,y)
    def step(self,ids,y):
        for b,group in enumerate(self.islands):
            ii=self.bootstrap_ids(ids,b)
            for island in group:island.step(ii,y)
        self.blocks+=1
    def diagnostics(self):
        p=np.array([[x.champion() for x in g] for g in self.islands])
        w=np.mean(np.var(p,axis=1,ddof=1),axis=0)
        between=np.var(p.mean(axis=1),axis=0,ddof=1)
        s=w/self.c.restarts
        d=np.maximum(between-s,0.)
        return {'predictions':p,'search':s,'data':d,'between':between,'total':np.var(p.reshape(-1,len(self.X)),axis=0,ddof=1)}
    def select(self,ids,y):
        # Pool all evolved candidates; final affine fitting uses acquired labels only.
        best=None
        for group in self.islands:
            for island in group:
                a,b,m,sc=affine_score(island.pred,ids,y,node_count(island.ops),self.c.penalty)
                i=int(np.argmin(sc))
                if best is None or sc[i]<best[0]:
                    best=(float(sc[i]),island.ops[i].copy(),island.constants[i].copy(),float(a[i]),float(b[i]))
        return best
    def predict(self,model,X):return model[3]+model[4]*evaluate(model[1][None,:],model[2][None,:],X)[0]
    def work(self):return sum(i.work for g in self.islands for i in g)

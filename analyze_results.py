#!/usr/bin/env python3
"""Reproduce numerical summaries from the saved independent experiment runs."""
from pathlib import Path
import argparse, json, hashlib, datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent
EPS=1e-8
ORDER=['adaptive','search_only','sample_always','sample_random','sample_maximin','fixed_25','fixed_50','fixed_75','uncorrected','adaptive_random','fixed_50_corrected','resample_only','dev_fixed']
NAMES={'adaptive':'Split','search_only':'Search','sample_always':'QBC-Every','sample_random':'Uniform-Every','sample_maximin':'Maximin-Every','fixed_25':'Fixed25','fixed_50':'Fixed50','fixed_75':'Fixed75','uncorrected':'Uncorrected','adaptive_random':'Split-Random','fixed_50_corrected':'Fixed50-Corrected','dev_fixed':'Dev-Fixed','resample_only':'Refresh-Only'}

def load(directory):
 rows=[];traces=[]
 for p in sorted(directory.glob('*.json')):
  x=json.loads(p.read_text())
  if 'spec' not in x or 'final' not in x:continue
  spec=x['spec'];final=x['final'];cfg=x['config'];n0=spec.get('initial_labels',16);q=len(x['acquired_indices'])-n0;blocks=final['blocks']-spec.get('warmup',2)
  assert len(set(x['acquired_indices']))==n0+q,p.name
  assert final['labels']==n0+q,p.name
  assert np.isclose(final['cost'],blocks+spec['query_cost']*q),p.name
  assert -1e-8<=spec.get('budget',48)-final['cost']<1+1e-8,p.name
  assert final['work']==cfg['bootstraps']*cfg['restarts']*cfg['population']*(1+cfg['generations']*final['blocks']),p.name
  assert np.isfinite(final['nmse']) and np.isfinite(final['ood_nmse']),p.name
  assert final['nmse']>=0 and final['ood_nmse']>=0,p.name
  row={**{k:v for k,v in x['spec'].items() if k!='engine'},**x['final'],'elapsed_seconds':x['elapsed_seconds'],'file':p.name}
  row['log_nmse']=np.log10(row['nmse']+EPS);row['log_ood_nmse']=np.log10(row['ood_nmse']+EPS)
  row['queries']=row['labels']-x['spec'].get('initial_labels',16)
  rows.append(row)
  for j,t in enumerate(x['trace']):traces.append({**{k:row[k] for k in ['problem','policy','seed','noise','query_cost']},'checkpoint':j,**t,'log_nmse':np.log10(t['nmse']+EPS)})
 return pd.DataFrame(rows),pd.DataFrame(traces)

def select_development(base,out):
 d,_=load(base/'development')
 assert len(d)==2376, f'Complete development set required: {len(d)}/2376'
 candidates=['search_only','fixed_25','fixed_50','fixed_75','sample_always']
 scores=d[d.policy.isin(candidates)].groupby(['query_cost','policy']).log_nmse.mean()
 selected={str(float(c)):scores.loc[c].idxmin() for c in sorted(d.query_cost.unique())}
 digest=hashlib.sha256(''.join(p.name+hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((base/'development').glob('*.json'))).encode()).hexdigest()
 result={'rule':'Minimum mean log10(NMSE+1e-8) on all development functions, seeds, and noise levels; separately by label price.','selected':selected,'development_sha256':digest,'scores':[{**dict(zip(['query_cost','policy'],i)),'mean_log10_nmse':float(v)} for i,v in scores.items()]}
 path=out/'development_selection.json'
 if path.exists():
  assert json.loads(path.read_text())==result,'Development selection differs from frozen file.'
 else:path.write_text(json.dumps(result,indent=2))
 return selected

def add_selected(d,selected):
 copy=d[[p==selected[str(float(c))] for p,c in zip(d.policy,d.query_cost)]].copy();copy['policy']='dev_fixed'
 return pd.concat([d,copy],ignore_index=True)

def paired(d,reference='adaptive',cluster='problem',replicates=10000):
 key=['problem','seed','noise','query_cost'];rows=[]
 for outcome in ['log_nmse','log_ood_nmse']:
  pivot=d.pivot(index=key,columns='policy',values=outcome)
  for p in pivot.columns:
   if p==reference:continue
   z=(pivot[reference]-pivot[p]).dropna().reset_index(name='difference')
   means=z.groupby(cluster).difference.mean().to_numpy()
   rng=np.random.default_rng(24817)
   draws=means[rng.integers(0,len(means),(replicates,len(means)))].mean(axis=1)
   lo,hi=np.quantile(10**draws,[.025,.975])
   rows.append({'reference':reference,'comparator':p,'outcome':outcome,'paired_runs':len(z),'clusters':len(means),'ratio':float(10**means.mean()),'lower':float(lo),'upper':float(hi),'win_fraction':float((z.difference<0).mean()),'tie_fraction':float((abs(z.difference)<1e-12).mean())})
 return pd.DataFrame(rows)

def summarize(d,groups):
 return d.groupby(groups).agg(runs=('nmse','size'),mean_log_nmse=('log_nmse','mean'),mean_log_expanded=('log_ood_nmse','mean'),median_nmse=('nmse','median'),mean_labels=('labels','mean'),mean_queries=('queries','mean'),mean_work=('work','mean'),mean_cost=('cost','mean'),mean_seconds=('elapsed_seconds','mean')).assign(gm_nmse=lambda x:10**x.mean_log_nmse,gm_expanded=lambda x:10**x.mean_log_expanded).reset_index()

def audit_summary(base,out):
 rows=[]
 for path in sorted((base/'audit').glob('*.json')):
  x=json.loads(path.read_text())
  for stage,s in enumerate(x['states']):
   risk=np.array(s['branch_nmse']).mean(axis=0);search,sample,refreshed=risk
   chosen=int(s['chosen_sample']);adv=np.log10((search+EPS)/(sample+EPS))
   gain=(search-sample)/max(search,sample,EPS)
   signal_den=s['data_signal']+2*s['search_signal'];sampling_score=s['data_signal']/signal_den if signal_den>0 else 0.
   rows.append({**x['spec'],'stage':stage,'spent':s['spent'],'search_signal':s['search_signal'],'data_signal':s['data_signal'],'signal_log_ratio':np.log10((s['data_signal']+1e-16)/(2*s['search_signal']+1e-16)),'sampling_score':sampling_score,'chosen_sample':chosen,'search_nmse':search,'sample_nmse':sample,'refresh_nmse':refreshed,'query_refresh_log_ratio':np.log10((sample+EPS)/(refreshed+EPS)),'refresh_search_log_ratio':np.log10((refreshed+EPS)/(search+EPS)),'sample_advantage_log10':adv,'relative_advantage':gain,'prefer_sample':int(sample<search),'tie':abs(gain)<=.01,'agreement':int(chosen==int(sample<search)),'log_regret':np.log10((risk[chosen]+EPS)/(min(risk[:2])+EPS)),'regret_nmse':risk[chosen]-min(risk[:2]),'always_search_log_regret':np.log10((search+EPS)/(min(risk[:2])+EPS)),'always_sample_log_regret':np.log10((sample+EPS)/(min(risk[:2])+EPS))})
 a=pd.DataFrame(rows);assert len(a)==720,len(a);a.to_csv(out/'audit_states.csv',index=False)
 def metrics(z):
  non=z[~z.tie]
  recalls=[non[non.prefer_sample==c].agreement.mean() for c in [0,1]]
  rho,p=spearmanr(z.sampling_score,z.relative_advantage)
  return {'states':len(z),'chosen_sample_fraction':float(z.chosen_sample.mean()),'sample_better_fraction':float(z.prefer_sample.mean()),'practical_tie_fraction':float(z.tie.mean()),'agreement_all':float(z.agreement.mean()),'agreement_non_ties':float(non.agreement.mean()),'balanced_accuracy_non_ties':float(np.mean(recalls)),'spearman':float(rho),'mean_log_regret':float(z.log_regret.mean()),'gm_risk_ratio_to_branch_oracle':float(10**z.log_regret.mean()),'mean_regret_nmse':float(z.regret_nmse.mean()),'always_search_gm_risk_ratio_to_branch_oracle':float(10**z.always_search_log_regret.mean()),'always_sample_gm_risk_ratio_to_branch_oracle':float(10**z.always_sample_log_regret.mean()),'chosen_search_gm_ratio':float(10**(z.log_regret-z.always_search_log_regret).mean()),'chosen_sample_gm_ratio':float(10**(z.log_regret-z.always_sample_log_regret).mean()),'query_refresh_gm_ratio':float(10**z.query_refresh_log_ratio.mean()),'refresh_search_gm_ratio':float(10**z.refresh_search_log_ratio.mean()),'query_better_than_refresh_fraction':float((z.sample_nmse<z.refresh_nmse).mean())}
 report={'overall':metrics(a),'by_stage':{str(k):metrics(z) for k,z in a.groupby('stage')},'by_noise':{str(k):metrics(z) for k,z in a.groupby('noise')}}
 # Equation-level bootstrap preserves states, seeds, and noise pairing.
 rng=np.random.default_rng(24818);clusters=sorted(a.problem.unique());boot=[]
 grouped=[a[a.problem==p] for p in clusters]
 for inds in rng.integers(0,len(clusters),(10000,len(clusters))):
  # Ratio estimators require retaining numerator and denominator, not averaging percentages.
  z=pd.concat([grouped[i] for i in inds],ignore_index=True);non=z[~z.tie]
  recalls=[non[non.prefer_sample==c].agreement.mean() for c in [0,1]]
  boot.append([z.agreement.mean(),np.mean(recalls),10**z.log_regret.mean(),10**(z.log_regret-z.always_search_log_regret).mean(),10**(z.log_regret-z.always_sample_log_regret).mean(),10**z.query_refresh_log_ratio.mean(),10**z.refresh_search_log_ratio.mean()])
 lo,hi=np.quantile(boot,[.025,.975],axis=0)
 report['equation_bootstrap_95']={k:[float(l),float(h)] for k,l,h in zip(['agreement_all','balanced_accuracy_non_ties','gm_risk_ratio_to_branch_oracle','chosen_search_gm_ratio','chosen_sample_gm_ratio','query_refresh_gm_ratio','refresh_search_gm_ratio'],lo,hi)}
 (out/'audit_summary.json').write_text(json.dumps(report,indent=2))
 return report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=ROOT/'results');ap.add_argument('--out',type=Path,default=ROOT/'analysis');ap.add_argument('--select-development',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 selected=select_development(a.results,a.out)
 if a.select_development:print(json.dumps(selected));return
 expected={'main':17280,'engineering':2160,'sensitivity_large':800,'sensitivity_b2r2':160,'sensitivity_b3r5':160,'sensitivity_b5r3':160}
 inventory={}
 for name,total in expected.items():
  d,t=load(a.results/name);assert len(d)==total,f'{name}: {len(d)}/{total}'
  inventory[name]=len(d)
  if name in ['main','engineering']:d=add_selected(d,selected);t=add_selected(t,selected)
  d.to_csv(a.out/f'{name}_runs.csv',index=False)
  summarize(d,['policy']).to_csv(a.out/f'{name}_overall.csv',index=False)
  summarize(d,['problem','noise','query_cost','policy']).to_csv(a.out/f'{name}_by_task.csv',index=False)
  summarize(d,['noise','query_cost','policy']).to_csv(a.out/f'{name}_by_regime.csv',index=False)
  paired(d).to_csv(a.out/f'{name}_paired.csv',index=False)
  if name in ['main','engineering']:
   paired(d[d.policy.isin(['fixed_50_corrected','fixed_50'])],reference='fixed_50_corrected').to_csv(a.out/f'{name}_acquisition_component.csv',index=False)
  if name=='main':
   t.groupby(['noise','query_cost','policy','checkpoint']).agg(cost=('cost','mean'),mean_log_nmse=('log_nmse','mean'),labels=('labels','mean'),count=('nmse','size')).reset_index().to_csv(a.out/'main_recorded_checkpoints.csv',index=False)
   summarize(d,['noise','policy']).to_csv(a.out/'main_by_noise.csv',index=False)
   noise_pairs=[]
   for noise,z in d.groupby('noise'):
    q=paired(z);q['noise']=noise;noise_pairs.append(q)
   pd.concat(noise_pairs).to_csv(a.out/'main_paired_by_noise.csv',index=False)
   floor_rows=[]
   for floor in [1e-6,1e-8,1e-10]:
    zz=d.copy();zz['log_nmse']=np.log10(zz.nmse+floor);q=paired(zz);q=q[q.outcome=='log_nmse'];q['error_floor']=floor;floor_rows.append(q)
   pd.concat(floor_rows).to_csv(a.out/'main_floor_sensitivity.csv',index=False)
  if name=='engineering':
   cond=[]
   for (p,c),z in d.groupby(['problem','query_cost']):
    q=paired(z,cluster='seed');q['problem']=p;q['query_cost']=c;cond.append(q[q.outcome=='log_nmse'])
   pd.concat(cond).to_csv(a.out/'engineering_paired_by_condition.csv',index=False)
  print(name,total,flush=True)
 report=audit_summary(a.results,a.out)
 (a.out/'inventory.json').write_text(json.dumps(inventory,indent=2));print(json.dumps(report['overall'],indent=2))
if __name__=='__main__':main()

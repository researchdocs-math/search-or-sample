#!/usr/bin/env python3
"""Run every experiment into a fresh result directory, then recompute summaries."""
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=10);p.add_argument('--out-root',type=Path,default=ROOT/'reproduced');a=p.parse_args();dest=a.out_root.resolve();dest.mkdir(parents=True,exist_ok=True)
 def run(*args):subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)
 studies=[('development','development'),('main','main'),('main_acquisition_controls','main'),('main_resampling_control','main'),('engineering','engineering'),('engineering_acquisition_controls','engineering'),('engineering_resampling_control','engineering'),('sensitivity_large','sensitivity_large'),('sensitivity_b2r2','sensitivity_b2r2'),('sensitivity_b3r5','sensitivity_b3r5'),('sensitivity_b5r3','sensitivity_b5r3')]
 for config,name in studies:
  run('run_experiments.py','--config',ROOT/'configs'/f'{config}.json','--workers',a.workers,'--out',dest/'results'/name)
  if name=='development':run('analyze_results.py','--results',dest/'results','--out',dest/'analysis','--select-development')
 run('calibrate_diagnostic.py','--out',dest/'results/calibration.json')
 run('audit_actions.py','--workers',a.workers,'--out',dest/'results/audit')
 run('analyze_results.py','--results',dest/'results','--out',dest/'analysis')
 print('Completed:',dest)
if __name__=='__main__':main()

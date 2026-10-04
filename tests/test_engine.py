import unittest,copy
import numpy as np
from sos.engine import Config,Ensemble,evaluate,node_count,affine_score
from sos.experiment import decision,run_one

class EngineTests(unittest.TestCase):
 def test_arithmetic_and_affine(self):
  X=np.linspace(-1,1,31)[:,None];o=np.array([[10,12,1,1,1,0,0]],dtype=np.int16);c=np.zeros_like(o,float)
  p=evaluate(o,c,X);np.testing.assert_allclose(p[0],X[:,0]**2+X[:,0]);self.assertEqual(node_count(o)[0],5)
  y=2+3*p[0];a,b,m,s=affine_score(p,np.arange(len(y)),y,node_count(o),0)
  np.testing.assert_allclose([a[0],b[0]],[2,3]);self.assertLess(m[0],1e-20)
 def test_protected_division(self):
  o=np.array([[13,1,0]],dtype=np.int16);X=np.array([[2.],[3.]])
  np.testing.assert_array_equal(evaluate(o,np.zeros_like(o,float),X)[0],X[:,0])
 def test_hidden_labels_do_not_change_state(self):
  X=np.linspace(-1,1,48)[:,None];y=X[:,0]**3;ids=np.arange(12)
  a=Ensemble(X,42,Config(population=8,generations=1,depth=3));b=copy.deepcopy(a)
  yp=y.copy();yp[12:]=100000
  a.step(ids,y);b.step(ids,yp)
  np.testing.assert_array_equal(a.diagnostics()['predictions'],b.diagnostics()['predictions'])
  self.assertEqual(a.select(ids,y)[0],b.select(ids,yp)[0])
 def test_valid_mutated_trees(self):
  X=np.random.default_rng(1).normal(size=(48,3));e=Ensemble(X,42,Config(population=12,generations=2,depth=3))
  for i in range(4):e.step(np.arange(16),X[:,0]**2+X[:,1])
  for g in e.islands:
   for a in g:
    self.assertTrue(np.all(a.ops[:,7:]<10));self.assertTrue(np.isfinite(a.pred).all());self.assertTrue(np.all(node_count(a.ops)<=15))
 def test_correction_and_cost_gate(self):
  d={'search':np.ones(5),'data':np.ones(5)*2,'between':np.ones(5)*3,'total':np.ones(5)*4}
  self.assertTrue(decision('adaptive',d,np.arange(5),0.25,0,np.random.default_rng(0))[0])
  self.assertFalse(decision('adaptive',d,np.arange(5),4,0,np.random.default_rng(0))[0])
 def test_repeatability_budget_and_no_repeat_queries(self):
  spec={'problem':'pilot_cubic','seed':123,'policy':'adaptive','query_cost':0.25,'budget':8,'engine':{'population':8,'generations':1,'depth':3}}
  a=run_one(spec);b=run_one(spec)
  self.assertEqual(a['model'],b['model']);self.assertEqual(a['acquired_indices'],b['acquired_indices']);self.assertEqual(a['final']['nmse'],b['final']['nmse'])
  self.assertLessEqual(a['final']['cost'],8);self.assertEqual(len(set(a['acquired_indices'])),len(a['acquired_indices']))
 def test_all_policy_resource_accounts(self):
  policies=['adaptive','search_only','sample_always','sample_random','sample_maximin','fixed_25','fixed_50','fixed_75','uncorrected','adaptive_random','fixed_50_corrected','resample_only']
  for policy in policies:
   for price in [.25,1.,4.]:
    with self.subTest(policy=policy,price=price):
     cfg={'population':4,'generations':1,'depth':2,'bootstraps':2,'restarts':2}
     z=run_one({'problem':'pilot_cubic','seed':129,'noise':.1,'policy':policy,'query_cost':price,'budget':5,'engine':cfg})
     q=len(z['acquired_indices'])-16;t=z['final']['blocks']-2
     self.assertAlmostEqual(z['final']['cost'],t+price*q)
     self.assertLessEqual(t+price*q,5)
     self.assertLess(5-(t+price*q),1)
     self.assertEqual(z['final']['work'],16*(1+z['final']['blocks']))
     self.assertEqual(len(set(z['acquired_indices'])),16+q)
if __name__=='__main__':unittest.main()

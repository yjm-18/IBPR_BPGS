import unittest,sys,ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ibpr_bpgs.metrics import THRESHOLDS,pooled_ap,lookup_for,gain,propagate
from ibpr_bpgs.targets import compact_cache,marginal_targets
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'auxiliary/boundary'))
from boundary_integrity import decode_softgroup_voxel_point_table,BoundaryIntegrityError
class Regressions(unittest.TestCase):
 def test_exact_decimal_thresholds(self):
  for t in THRESHOLDS:
   e={'iou':np.array([[t]]),'scores':np.array([.9])}
   self.assertEqual(pooled_ap([e],t),1)
   self.assertEqual(pooled_ap([e],np.nextafter(t,np.inf)),0)
  np.testing.assert_array_equal(THRESHOLDS,[.5,.55,.6,.65,.7,.75,.8,.85,.9,.95])
 def test_reappearing_source_label(self):
  src=np.array([1,1,1,2,2]);nn=np.array([[0,1,2,3,4]]);gt=np.array([1]);base=propagate(src,nn)
  after=propagate(np.array([2,1,1,2,2]),nn);lookup=lookup_for(gt,base,src)
  self.assertEqual(lookup[2],-1);self.assertEqual(gain(gt,base,after,lookup)['H'],1)
  d=dict(chosen=np.array([True]),ids=np.array([0]),labels=np.array([0]),dest=np.array([2]),components=np.array([0]))
  cache=compact_cache(SimpleNamespace(pred=src),gt,nn,base,d)
  np.testing.assert_array_equal(marginal_targets(cache,1),[-1])
 def test_count_prefixed_rulebook(self):
  inv=np.array([0,0,1]);table=np.array([[2,0,1],[1,2,0]])
  decoded=decode_softgroup_voxel_point_table(table,inv,3,2)
  np.testing.assert_array_equal(decoded,[[0,1],[2,-1]])
  feats=np.array([[2.,4.],[4.,8.],[9.,3.]])
  means=np.stack([feats[row[row>=0]].mean(0) for row in decoded])
  np.testing.assert_array_equal(means,[[3.,6.],[9.,3.]])
  with self.assertRaises(BoundaryIntegrityError):decode_softgroup_voxel_point_table(np.array([[2,0,0],[1,2,0]]),inv,3,2)
  with self.assertRaises(BoundaryIntegrityError):decode_softgroup_voxel_point_table(np.array([[3,0,1],[1,2,0]]),inv,3,2)
 def test_grouped_boundary_ap(self):
  tree=ast.parse((ROOT/'auxiliary/boundary/train_boundary_only.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BoundaryMetrics');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compute_average_precision');env={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'AP','exec'),env)
  f=env['compute_average_precision']
  for labels in [[1,0],[0,1]]:self.assertEqual(f(SimpleNamespace(all_probs=[.5,.5],all_labels=labels)),.5)
  from sklearn.metrics import average_precision_score
  rng=np.random.default_rng(42);p=rng.integers(0,6,100)/5;y=rng.integers(0,2,100)
  self.assertAlmostEqual(f(SimpleNamespace(all_probs=p,all_labels=y)),average_precision_score(y,p))
if __name__=='__main__':unittest.main()

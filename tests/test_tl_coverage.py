import unittest
from types import SimpleNamespace
import numpy as np
from ibpr_bpgs.regions import region_data, FEATURE_NAMES

class TLCoverageTests(unittest.TestCase):
    def example(self, n):
        pos=np.zeros((n+2,3),np.float32)
        pos[:,0]=np.r_[np.arange(n)*0.001,100,101]
        scene=SimpleNamespace(pos=pos,pred=np.r_[np.ones(n,np.int32),[2,2]].astype(np.int32),
            predicted_tree=np.ones(n+2,bool),offset=np.zeros_like(pos),
            embedding=np.zeros((n+2,5),np.float32),tree_probability=np.ones(n+2,np.float32),
            prediction_count=np.ones(n+2,np.float32),boundary_score=np.zeros(n+2,np.float32))
        scores=dict(point_ids=np.arange(n),candidates=np.tile([1,2,0,0],(n,1)),
                    scores=np.tile(np.array([.1,.9,-1,-1],np.float32),(n,1)))
        return scene,scores

    def test_constant_for_actual_regions_of_different_sizes(self):
        column=FEATURE_NAMES.index('log_prediction_count_mean')
        for n in [1,15,54,1000]:
            scene,scores=self.example(n)
            cfg=dict(score=.4,margin=.02)
            tl=region_data(scene,scores,cfg,model='TL')
            old=region_data(scene,scores,cfg)
            self.assertEqual(len(tl['components']),1)
            self.assertEqual(tl['features'][0,column],np.float32(np.log(2)))
            mask=np.arange(38)!=column
            np.testing.assert_array_equal(tl['features'][:,mask],old['features'][:,mask])
            sat=region_data(scene,scores,cfg,model='SAT')
            np.testing.assert_array_equal(sat['features'],old['features'])

    def test_sat_keeps_variable_coverage_and_tl_rejects_it(self):
        scene,scores=self.example(54);scene.prediction_count[::2]=3
        cfg=dict(score=.4,margin=.02)
        sat=region_data(scene,scores,cfg,model='SAT')
        self.assertEqual(sat['features'][0,22],np.log1p(scene.prediction_count[:54]).mean())
        with self.assertRaises(ValueError):region_data(scene,scores,cfg,model='TL')

if __name__=='__main__': unittest.main()

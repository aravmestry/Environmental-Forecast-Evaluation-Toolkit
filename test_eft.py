import unittest
import numpy as np
import pandas as pd
from eft import evaluate, temporal_holdout, cluster_bootstrap_delta

class ToolkitTests(unittest.TestCase):
    def test_perfect_predictions(self):
        d=evaluate([0,1,0,1], [0,1,0,1])
        self.assertEqual(d['auc'],1)
        self.assertEqual(d['brier_score'],0)
    def test_single_class_auc_undefined(self):
        self.assertIsNone(evaluate([0,0],[0.1,0.2])['auc'])
    def test_invalid_probabilities(self):
        with self.assertRaises(ValueError): evaluate([0,1],[0.5,1.2])
    def test_chronological_holdout(self):
        frame=pd.DataFrame({'date':['2023-12-31','2024-01-01','2024-01-02'],'x':[1,2,3]})
        a,b=temporal_holdout(frame,'date','2024-01-01')
        self.assertEqual((len(a),len(b)),(1,2))
    def test_cluster_bootstrap_deterministic(self):
        df=pd.DataFrame({'site':np.repeat(list('ABCD'), 6),'event':[0,1,0,1,0,1]*4,
             'baseline':[.4,.6,.4,.6,.4,.6]*4,'candidate':[.1,.9,.1,.9,.1,.9]*4})
        a=cluster_bootstrap_delta(df,n_boot=40,seed=123)
        b=cluster_bootstrap_delta(df,n_boot=40,seed=123)
        self.assertEqual(a,b)
        self.assertLess(a['metrics']['brier_score']['delta'],0)
    def test_missing_cluster(self):
        with self.assertRaises(ValueError):
            cluster_bootstrap_delta(pd.DataFrame({'site':['a','a'],'event':[0,1],'baseline':[.2,.6],'candidate':[.1,.9]}))

if __name__=='__main__': unittest.main()

import os
os.chdir('../')
import numpy as np
import unittest
from src import synthetic_data, detect_anomaly
from sklearn.ensemble import IsolationForest

class test_generate_data(unittest.TestCase):
    def setUp(self):
        self.n_cluster = 3
        self.n_features = 3
        self.mean = np.array([[0,2,-1],[-1,0,1],[1,-2,0]])
        self.cov = np.repeat([0.01*np.ones([self.n_features, self.n_features])], self.n_cluster) \
            .reshape(3,3,3)
    
    def test_generate(self):
        synth_data = synthetic_data.generateClusters(self.n_cluster, self.mean, self.cov, self.n_features)
        X,y = synth_data.generate_data()
        self.assertEqual(X.shape, (100,3,3))
        self.assertListEqual([[0,2,-1],[-1,0,1],[1,-2,0]], X.mean(axis=0).round(0).tolist())

    def test_isolationforest(self):
        synth_data = synthetic_data.generateClusters(self.n_cluster, self.mean, self.cov, self.n_features)
        X,y = synth_data.generate_data()
        clf = detect_anomaly.runIsolationForest()
        pred = clf.fit_predict(X_train=X[:,:,0], \
                                X_test=np.array([0,-1,1]).reshape(1,3))
        self.assertEqual(list(pred),[1])
        # pred = clf.fit_predict(X_train=X[:,1].reshape(-1,1), \
        #                         X_test=np.array([-1,-2,1,2,0]).reshape(-1,1))
        # self.assertEqual(list(pred),[1,-1,-1,-1,-1])
        # pred = clf.fit_predict(X_train=X[:,2].reshape(-1,1), \
        #                         X_test=np.array([-1,-2,1,2,0]).reshape(-1,1))
        # self.assertEqual(list(pred),[-1,-1,1,-1,-1])

        
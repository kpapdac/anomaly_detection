import os
print(os.getcwd())
#os.chdir('../')
import numpy as np
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from anomaly_detection import loader, synthetic_data, detect_anomaly

class test_bnn(unittest.TestCase):
    def setUp(self):
        self.n_cluster = 3
        self.n_features = 3000
        self.mean = np.random.normal(0,2,[self.n_features,self.n_cluster])
        self.cov = np.repeat([0.01*np.ones([self.n_features, self.n_features])], self.n_cluster) \
            .reshape(self.n_features,self.n_features,self.n_cluster)

    def test_bnn(self):
        synth_data = synthetic_data.generateClusters(self.n_cluster, self.mean, self.cov, self.n_features)
        X,y = synth_data.generate_data()
        bnn_model = detect_anomaly.Model()
        x = torch.from_numpy(X[:,:,0]).type(torch.float)
        y = torch.from_numpy(np.repeat(0,100))
        opt = detect_anomaly.optimizeBNN(x, y, bnn_model)
        opt.train_loop()
import os
os.chdir('../')
import numpy as np
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from anomaly_detection import loader, synthetic_data, detect_anomaly

class test_nn(unittest.TestCase):
    def setUp(self):
        self.n_cluster = 3
        self.n_features = 3000
        self.mean = np.random.normal(0,2,[self.n_features,self.n_cluster])
        self.cov = np.repeat([0.01*np.ones([self.n_features, self.n_features])], self.n_cluster) \
            .reshape(self.n_features,self.n_features,self.n_cluster)

    def test_twolayernn(self):
        synth_data = synthetic_data.generateClusters(self.n_cluster, self.mean, self.cov, self.n_features)
        X,y = synth_data.generate_data()
        nn_model = detect_anomaly.NeuralNetwork(D_in=self.n_features,H=30)
        x = torch.from_numpy(X[:,:,0]).type(torch.float)
        y = torch.from_numpy(np.repeat(0,100))
        # x = torch.randn(128, 20)
        # y = torch.from_numpy(np.repeat(0,128))
        X_y = [(x[i],y[i]) for i in range(len(x))]
        dataset_ = loader.FeatureClusterDataset(X_y)
        train_dataloader = DataLoader(dataset_, batch_size=64)
        opt = detect_anomaly.optimizeNN(train_dataloader, nn_model, learning_rate=1e-3, batch_size=64, epochs=5)
        opt.train_loop()
        opt.iterate()
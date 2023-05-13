import os
os.chdir('../')
import numpy as np
import unittest
import torch
import torch.nn as nn
from pyro.nn.module import PyroModule, PyroParam, PyroSample
from pyro.contrib.bnn import HiddenLayer

class test_bnn(unittest.TestCase):
    def setUp(self):
        self.X = torch.Tensor([0,1])
        self.A_mean = torch.Tensor([0,0])
        self.A_scale = torch.Tensor([[1,1],[1,1]])
        self.bnet = HiddenLayer(X=self.X, A_mean=self.A_mean, A_scale=self.A_scale)
        
    def test_sample(self):
        print(self.bnet.log_prob)
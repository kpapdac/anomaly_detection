import numpy as np
import pandas as pd
import sklearn

class generateClusters:
    def __init__(self, n_cluster: int, mean: list, cov: list, n_features: int):
        self.n_cluster = n_cluster
        self.mean = mean
        self.cov = cov
        self.n_features = n_features

    def generate_data(self):
        X = np.zeros([100, self.n_features, self.n_cluster])
        y = np.zeros([100, self.n_cluster])
        for cl in range(self.n_cluster):
            X[:,:,cl] = np.random.multivariate_normal(self.mean[:,cl], self.cov[:,:,cl], 100)
            y[:,cl] = cl
        return X, y #np.vstack(X), np.hstack(y).reshape(-1,1)


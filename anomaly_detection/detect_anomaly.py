from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn import mixture
import torch.nn as nn
import torch
import torch.nn.functional as F

class runIsolationForest:
    def __init__(self, max_samples=100):
        self.max_samples = max_samples
    
    def fit_predict(self, X_train, X_test):
        clf = IsolationForest(max_samples=self.max_samples, random_state=0)
        clf.fit(X_train)
        return clf.predict(X_test)

class runOneClassSVM:
    def __init__(self, kernel='rbf', gamma='scale'):
        self.kernel = kernel
        self.gamma = gamma
    
    def fit_predict(self, X_train, X_test):
        clf = OneClassSVM(kernel=self.kernel, gamma=self.gamma)
        clf.fit(X_train)
        return clf.predict(X_test)

class runKMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def fit_predict(self, X_train, X_test):
        clf = KMeans(n_clusters=self.n_clusters)
        clf.fit(X_train)
        return clf.predict(X_test)

class runGaussianMixture:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def fit_predict(self, X_train, X_test):
        gmm = mixture.GaussianMixture(n_components=self.n_clusters, covariance_type="full")
        gmm.fit(X_train)
        return gmm.predict(X_test), gmm.means_, gmm.covariances_

class NeuralNetwork(nn.Module):
    def __init__(self, D_in, H):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Linear(D_in, H)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class optimizeNN():
    def __init__(self, dataloader, model, learning_rate, batch_size, epochs):
        self.dataloader = dataloader
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train_loop(self):
        size = len(self.dataloader.dataset)
        for batch, (X, y) in enumerate(self.dataloader):
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def iterate(self):        
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_loop()
        print("Done!")
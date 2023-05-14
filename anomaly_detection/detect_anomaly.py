from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn import mixture
import torch.nn as nn
import torch
from functools import partial
import pyro
from pyro.nn import PyroModule, PyroSample
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm.auto import trange, tqdm
from pyro.infer.autoguide import AutoDiagonalNormal


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

class optimizeNN:
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
            print(pred[0,:])
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

class Model(PyroModule):
    def __init__(self, h1=20, h2=20):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](3000, h1)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([h1, 3000]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([h1]).to_event(1))
        self.fc2 = PyroModule[nn.Linear](h1, h2)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([h2, h1]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([h2]).to_event(1))
        self.fc3 = PyroModule[nn.Linear](h2, 1)
        self.fc3.weight = PyroSample(dist.Normal(0., 1.).expand([1, h2]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        #x = x.reshape(-1, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.fc3(x).squeeze()
        sigma = pyro.sample("sigma", dist.Uniform(0., 1.))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(torch.ones(1)), obs=y)
        return obs

class optimizeBNN:
    def __init__(self, x, y, model):
        self.x = x
        self.y = y
        self.model = model
        self.guide = AutoDiagonalNormal(self.model)
        adam = pyro.optim.Adam({"lr": 1e-3})
        self.svi = SVI(self.model, self.guide, adam, loss=Trace_ELBO())

    def train_loop(self):
        pyro.clear_param_store()
        bar = trange(2000)
        for epoch in bar:
            loss = self.svi.step(self.x, self.y)
            bar.set_postfix(loss=f'{loss / self.x.shape[0]:.3f}')
            output = self.model(self.x)
            correct = (output == self.y).float().sum()
            accuracy = 100 * correct / len(self.y)
            print("Accuracy = {}".format(accuracy))
        print("Done!")

# n_cluster = 3
# n_features = 3000
# mean = np.random.normal(0,2,[n_features,n_cluster])
# cov = np.repeat([0.01*np.ones([n_features, n_features])], n_cluster) \
#     .reshape(n_features,n_features,n_cluster)
# synth_data = synthetic_data.generateClusters(n_cluster, mean, cov, n_features)
# X,y = synth_data.generate_data()
# x = torch.from_numpy(X[:,:,0]).type(torch.float)
# y = torch.from_numpy(np.repeat(0,100))
# X_y = [(x[i],y[i]) for i in range(len(x))]
# dataset_ = loader.FeatureClusterDataset(X_y)
# train_dataloader = DataLoader(dataset_, batch_size=64)
# model = Model()
# guide = AutoDiagonalNormal(model)
# adam = pyro.optim.Adam({"lr": 1e-3})
# svi = SVI(model, guide, adam, loss=Trace_ELBO())

# pyro.clear_param_store()
# bar = trange(2000)
# correct = 0
# for epoch in bar:
#     loss = svi.step(x, y)
#     bar.set_postfix(loss=f'{loss / x.shape[0]:.3f}')
#     output = model(x)
#     correct = (output == y).float().sum()

#     accuracy = 100 * correct / len(y)
#     print("Accuracy = {}".format(accuracy))
# predictive = Predictive(model, guide=guide, num_samples=500)
# preds = predictive(x)

# y_pred = preds['obs'].T.detach().numpy().mean(axis=1)
# y_std = preds['obs'].T.detach().numpy().std(axis=1)

# print('DD')
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn import mixture
import torch.nn as nn
import torch
from functools import partial
import pyro
import pyro.contrib.bnn as bnn
from pyro import poutine
from torch.distributions import constraints
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import tyxe
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import loader, synthetic_data
import pyro.optim as pyroopt
from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO
from tqdm.auto import trange, tqdm

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

class BNN(nn.Module):
    def __init__(self, n_hidden=1024, n_classes=30):
        super(BNN, self).__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
       
    def model(self, images, labels=None, kl_factor=1.0): 
        n_images = images.size(0)
        # Set-up parameters for the distribution of weights for each layer `a<n>`
        a1_mean = torch.zeros(3000, self.n_classes)
        a1_scale = torch.ones(3000, self.n_classes) 
        a1_dropout = torch.tensor(0.25)
        # Mark batched calculations to be conditionally independent given parameters using `plate`
        with pyro.plate('data', size=n_images):
            # Sample first hidden layer
            logits = pyro.sample('logits', bnn.HiddenLayer(images, a1_mean, a1_scale,
                                                           non_linearity=lambda x: F.log_softmax(x, dim=-1),
                                                           KL_factor=kl_factor,
                                                           include_hidden_bias=False))
            # One-hot encode labels
            labels = F.one_hot(labels) if labels is not None else None
            # Condition on observed labels, so it calculates the log-likehood loss when training using VI
            return pyro.sample('label', torch.distributions.OneHotCategorical(logits=logits), obs=labels) 
    
    def guide(self, images, labels=None, kl_factor=1.0):
        #images = images.view(-1, 784)
        n_images = images.size(0)
        # Set-up parameters to be optimized to approximate the true posterior
        # Mean parameters are randomly initialized to small values around 0, and scale parameters
        # are initialized to be 0.1 to be closer to the expected posterior value which we assume is stronger than
        # the prior scale of 1.
        # Scale parameters must be positive, so we constraint them to be larger than some epsilon value (0.01).
        # Variational dropout are initialized as in the prior model, and constrained to be between 0.1 and 1 (so dropout
        # rate is between 0.1 and 0.5) as suggested in the local reparametrization paper
        a1_mean = pyro.param('a1_mean', 0.01 * torch.randn(3000, self.n_classes))
        a1_scale = pyro.param('a1_scale', 0.1 * torch.ones(3000, self.n_classes),
                              constraint=constraints.greater_than(0.01))
        a1_dropout = pyro.param('a1_dropout', torch.tensor(0.25),
                                constraint=constraints.interval(0.1, 1.0))

        # Sample latent values using the variational parameters that are set-up above.
        # Notice how there is no conditioning on labels in the guide!
        with pyro.plate('data', size=n_images):
           logits = pyro.sample('logits', bnn.HiddenLayer(images, a1_mean, a1_scale,
                                                           non_linearity=lambda x: F.log_softmax(x, dim=-1),
                                                           KL_factor=kl_factor,
                                                           include_hidden_bias=False))
    
    def infer_parameters(self, loader, lr=0.01, momentum=0.9,
                         num_epochs=30):
        optim = pyroopt.SGD({'lr': lr, 'momentum': momentum, 'nesterov': True})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self.model, self.guide, optim, elbo)
        kl_factor = loader.batch_size / len(loader.dataset)
        for i in range(num_epochs):
            total_loss = 0.0 
            total = 0.0
            correct = 0.0
            for images, labels in loader:
                loss = svi.step(images, labels, kl_factor=kl_factor)
                pred = self.forward(images, n_samples=1).mean(0) 
                total_loss += loss / len(loader.dataset)
                total += labels.size(0)
                correct += (pred.argmax(-1) == labels).sum().item()
                param_store = pyro.get_param_store()
            print(f"[Epoch {i + 1}] loss: {total_loss:.5E} accuracy: {correct / total * 100:.5f}")

    def forward(self, images, n_samples=10):
        res = []
        for i in range(n_samples):
            t = poutine.trace(self.guide).get_trace(images)
            res.append(t.nodes['logits']['value'])
        return torch.stack(res, dim=0) 

class Model(PyroModule):
    def __init__(self, h1=20, h2=20):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](1, h1)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([h1, 1]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([h1]).to_event(1))
        self.fc2 = PyroModule[nn.Linear](h1, h2)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([h2, h1]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([h2]).to_event(1))
        self.fc3 = PyroModule[nn.Linear](h2, 30)
        self.fc3.weight = PyroSample(dist.Normal(0., 1.).expand([1, h2]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))
        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        x = x.reshape(-1, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.fc3(x).squeeze()
        sigma = pyro.sample("sigma", dist.Uniform(0., 1.))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(30), obs=y)
        return mu

n_cluster = 3
n_features = 3000
mean = np.random.normal(0,2,[n_features,n_cluster])
cov = np.repeat([0.01*np.ones([n_features, n_features])], n_cluster) \
    .reshape(n_features,n_features,n_cluster)
synth_data = synthetic_data.generateClusters(n_cluster, mean, cov, n_features)
X,y = synth_data.generate_data()
x = torch.from_numpy(X[:,:,0]).type(torch.float)
y = torch.from_numpy(np.repeat(0,100))
X_y = [(x[i],y[i]) for i in range(len(x))]
dataset_ = loader.FeatureClusterDataset(X_y)
train_dataloader = DataLoader(dataset_, batch_size=64)
bayesnn = BNN()
adam = pyro.optim.Adam({"lr": 1e-3})
svi = SVI(bayesnn.model, bayesnn.guide, adam, loss=Trace_ELBO())

# pyro.clear_param_store()
# optim = pyro.optim.Adam({"lr": 1e-3})
# elbos = []
# def callback(bnn, i, e):
#     elbos.append(e)
pyro.clear_param_store()
bar = trange(20000)
for epoch in bar:
    loss = svi.step(x, y)
    bar.set_postfix(loss=f'{loss / x.shape[0]:.3f}')

# with tyxe.poutine.local_reparameterization():
#     bayesnn.fit(train_dataloader, optim, 10000, callback)
print('DD')
# bayesnn.infer_parameters(train_dataloader, num_epochs=30, lr=0.002)
# net = nn.Sequential(nn.Linear(3000, 30))
# prior = tyxe.priors.IIDPrior(torch.distributions.Normal(0, 1))
# obs_model = tyxe.likelihoods.Categorical(len(x))
# guide = partial(tyxe.guides.AutoNormal, init_scale=0.01)
# bnn = tyxe.VariationalBNN(net, prior, obs_model, guide)

# pyro.clear_param_store()
# optim = pyro.optim.Adam({"lr": 1e-3})
# elbos = []
# def callback(bnn, i, e):
#     elbos.append(e)
    
# with tyxe.poutine.local_reparameterization():
#     bnn.fit(train_dataloader, optim, 10000, callback)
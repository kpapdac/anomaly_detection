from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn import mixture

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

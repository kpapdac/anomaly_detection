from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
# gaussian mixture

class runIsolationForest:
    def __init__(self, max_samples=100):
        self.max_samples = max_samples
    
    def fit_predict(self, X_train, X_test):
        clf = IsolationForest(max_samples=self.max_samples, random_state=0)
        clf.fit(X_train)
        return clf.predict(X_test)

"""
Piecewise Linear Network
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from sklearn.cluster import KMeans
from scipy import linalg

class BasePLN(BaseEstimator):
    """ A base PLN class .

    Parameters
    ----------
    n_clusters : number of clusters in the PLN, optional
        Number of clusters.
    """
    def __init__(self, n_clusters=30):
        self.n_clusters = n_clusters
        self.random_state = 0

    def fit(self, X, y):
        """ Fit a PLN model

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        self.n_samples = X.shape[0]
        self.n_inputs = X.shape[1]
        if y.ndim==1:
            self.n_outputs = 1
        else:
            self.n_outputs = y.shape[1]
        if self.n_outputs > 1:
            self.multi_output=True
        print(self.n_inputs)
        print(self.n_outputs)
        X, y = check_X_y(X, y, multi_output=self.multi_output, ensure_min_samples=self.n_samples)

        k_means = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(X)
        self.cluster_centers = k_means.cluster_centers_
        self.labels = k_means.labels_

        self.R = np.empty((self.n_clusters,self.n_inputs+1,self.n_inputs+1))
        self.C = np.empty((self.n_clusters,self.n_inputs+1,self.n_outputs))
        self.W = np.empty((self.n_clusters,self.n_inputs+1,self.n_outputs))
        for k in np.arange(self.n_clusters):
            idxk = self.labels==k
            X_k = np.concatenate((np.ones((sum(idxk),1)), X[idxk,:]), axis=1)
            y_k = y[idxk].reshape(sum(idxk),self.n_outputs)
            self.R[k] = X_k.T @ X_k
            self.C[k] = X_k.T @ y_k
            self.W[k] = linalg.lstsq(self.R[k], self.C[k])[0]
            print(self.W[k])

        print(self.W)

        # Return the estimator
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            Returns :math:`x^2` where :math:`x` is the first column of `X`.
        """
        X = check_array(X)
        return X[:, 0]**2

if __name__ == "__main__":
    np.random.seed(0)
    n_samples = 50
    X = np.random.random((n_samples,3))
    y = np.random.random((n_samples,2))
    print(X)
    print(y)
    pln = BasePLN(5)
    model = pln.fit(X, y)
    print(model)

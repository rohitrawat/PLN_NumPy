"""
Piecewise Linear Network
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer

from sklearn.cluster import KMeans
from scipy import linalg

class BasePLN(BaseEstimator):
    """ A base PLN class .

    Parameters
    ----------
    n_clusters : number of clusters in the PLN, optional
        Number of clusters.
    """
    def __init__(self, n_clusters=None):
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
        else:
            self.multi_output=False

        if self.n_clusters is None:
            self.n_clusters = self._suggestNumClusters()

        print("PLN::fit() n_inputs: %(n_inputs)d, n_outputs: %(n_outputs)d, n_clusters: %(n_clusters)d" % self.__dict__)
        X, y = check_X_y(X, y, multi_output=self.multi_output, ensure_min_samples=self.n_clusters)

        self.k_means = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(X)
        self.cluster_centers = self.k_means.cluster_centers_
        self.labels = self.k_means.labels_

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
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            Returns : predicted output of the PLN.
        """
        print('predicting')
        # Check is fit had been called
        check_is_fitted(self, ['W', 'R', 'C'])

        X = check_array(X)
        pred_cluster_labels = self.k_means.predict(X)
        y = np.zeros((X.shape[0],self.n_outputs))
        for k in np.arange(self.n_clusters):
            idxk = pred_cluster_labels==k
            X_k = np.concatenate((np.ones((sum(idxk),1)), X[idxk,:]), axis=1)
            y_k = X_k @ self.W[k]
            y[idxk,:] = y_k
        return y

    def _suggestNumClusters(self):
        Nc_limit = np.ceil(self.n_samples / (self.n_inputs + 1))
        print("Memorization limits K to: %d" % Nc_limit)
        if Nc_limit < 1:
            Nc_limit = 1

        if self.n_samples>200 and self.n_samples<1000:
            Nc_limit = Nc_limit*2;
            print("But I'll allow: %d" % Nc_limit);
        elif self.n_samples>=1000:
            Nc_limit = Nc_limit*5 # rohit nov 7 2016, changes 2,5 to 4,10
            print("But I'll allow: %d" % Nc_limit)

        if self.n_samples <= 100:
            Nc_recommended = round(self.n_samples / 10);
        else:
            Nc_recommended = round(12 * log10(self.n_samples) - 11);

        if Nc_recommended < 1:
            Nc_recommended = 1;

        print("Rule of thumb value for K: %d" % Nc_recommended)

        if Nc_limit < Nc_recommended:
            Nc = Nc_limit
        else:
            Nc = Nc_recommended;

        print("Chosen value of K: %d" % Nc)

        return Nc

    def prune(self):
        pass

class PLNR(BasePLN, RegressorMixin):
    pass

class PLNC(BasePLN, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = unique_labels(y)

        self.lb = LabelBinarizer()
        y_ohe = self.lb.fit_transform(y)
        # print("y_ohe")
        # print(y_ohe)
        ret = super().fit(X, y_ohe)
        return ret

    def predict(self, X):
        y_ohe = super().predict(X)
        y = self.lb.inverse_transform(y_ohe)
        return y

def main():
    testClassifier()
    #testRegressor()

def testClassifier():
    np.random.seed(0)
    n_samples = 50
    X = np.random.random((n_samples,3))
    y = np.random.random_integers(1,5,n_samples)+2
    print("X:")
    print(X)
    print("y:")
    print(y)
    pln = PLNC()
    model = pln.fit(X, y)
    print(model._suggestNumClusters())
    y_pred = model.predict(X)
    print("pred_y:")
    print(y_pred)
    print("E:")
    print(sum(y == y_pred)/len(y))

    print("Starting CV on trained model..")
    from sklearn.model_selection import cross_val_score
    p = PLNR()
    print(cross_val_score(p, X, y.ravel(), cv=4, verbose=1))

def testRegressor():
    np.random.seed(0)
    n_samples = 50
    X = np.random.random((n_samples,3))
    y = np.random.random((n_samples,1))
    X = np.arange(n_samples).reshape((n_samples,1))
    y = np.sin(X)+np.random.random((n_samples,1))
    print("X:")
    print(X)
    print("y:")
    print(y)
    pln = PLNR()
    model = pln.fit(X, y)
    print(model._suggestNumClusters())
    y1 = model.predict(X)
    print("pred_y:")
    print(y1)
    print("E:")
    print(sum(y-y1))
    import matplotlib.pyplot as plt
    plt.plot(X,y)
    plt.draw()
    plt.figure()
    plt.plot(X,y1)
    plt.draw()

    print("Starting CV on trained model..")
    from sklearn.model_selection import cross_val_score
    p = PLNR()
    print(cross_val_score(p, X, y.ravel(), cv=4, verbose=1))

if __name__ == "__main__":
    main()

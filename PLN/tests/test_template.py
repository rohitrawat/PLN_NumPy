import numpy as np
from numpy.testing import assert_almost_equal

from PLN import BasePLN, PLNR, PLNC


# def test_demo():
#     X = np.random.random((100, 10))
#     estimator = TemplateEstimator()
#     estimator.fit(X, X[:, 0])
#     assert_almost_equal(estimator.predict(X), X[:, 0]**2)

def main():
    testClassifier()
    testRegressor()

def testClassifier():
    np.random.seed(0)
    n_samples = 50
    X = np.random.random((n_samples,3))
    y = np.random.randint(1,5+1,n_samples)+2
    # print("X:")
    # print(X)
    # print("y:")
    # print(y)
    pln = PLNC()
    model = pln.fit(X, y)
    y_pred = model.predict(X)
    # print("pred_y:")
    # print(y_pred)
    # print("E:")
    # print(sum(y == y_pred)/len(y))

    print("Starting CV on trained model..")
    from sklearn.model_selection import cross_val_score
    p = PLNR()
    #print(cross_val_score(p, X, y.ravel(), cv=4, verbose=1))

def testRegressor():
    np.random.seed(0)
    n_samples = 50
    X = np.random.random((n_samples,3))
    y = np.random.random((n_samples,1))
    X = np.arange(n_samples).reshape((n_samples,1))
    y = np.sin(X)+np.random.random((n_samples,1))
    # print("X:")
    # print(X)
    # print("y:")
    # print(y)
    pln = PLNR()
    model = pln.fit(X, y)
    y1 = model.predict(X)
    # print("pred_y:")
    # print(y1)
    # print("E:")
    # print(sum(y-y1))
    # import matplotlib.pyplot as plt
    # plt.plot(X,y)
    # plt.draw()
    # plt.figure()
    # plt.plot(X,y1)
    # plt.draw()

    print("Starting CV on trained model..")
    from sklearn.model_selection import cross_val_score
    p = PLNR()
    #print(cross_val_score(p, X, y.ravel(), cv=4, verbose=1))

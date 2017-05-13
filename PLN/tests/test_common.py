from sklearn.utils.estimator_checks import check_estimator
from PLN import BasePLN, PLNR, PLNC


def test_estimator():
    return check_estimator(PLNR)


def test_classifier():
    return check_estimator(PLNC)


def test_base():
    return check_estimator(BasePLN)

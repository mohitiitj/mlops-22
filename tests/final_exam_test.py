import sys, os
import numpy as np
from joblib import load

from mlops.utils import get_all_h_param_comb, tune_and_save
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

def helper_h_params():
    # small number of h params
    gamma_list = [0.01, 0.005]
    c_list = [0.1, 0.2]

    params = {}
    params["gamma"] = gamma_list
    params["C"] = c_list
    h_param_comb = get_all_h_param_comb(params)
    return h_param_comb

def helper_create_bin_data(n=100, d=7):
    x_train_0 = np.random.randn(n, d)
    x_train_1 = 1.5 + np.random.randn(n, d)
    x_train = np.vstack((x_train_0, x_train_1))
    y_train = np.zeros(2 * n)
    y_train[n:] = 1

    return x_train, y_train

def test_random_seed_with_integer():
    h_param_comb = helper_h_params()
    X, y = helper_create_bin_data(n=1000, d=7)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y,
                                   test_size=0.20, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y,
                                   test_size=0.20, random_state=42)
    assert (X_train1 == X_train2).all()
    assert (X_test1 == X_test2).all()
    assert (y_train1 == y_train2).all()
    assert (y_test1 == y_test2).all()


def test_random_seed_without_integer():
    h_param_comb = helper_h_params()
    X, y = helper_create_bin_data(n=1000, d=7)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y,
                                   test_size=0.20)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y,
                                   test_size=0.20)
    assert (X_train1 == X_train2).all() == False
    assert (X_test1 == X_test2).all() == False
    assert (y_train1 == y_train2).all() == False
    assert (y_test1 == y_test2).all() == False
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.special as spys
import math
import plot_results as pr
import pandas as pd


def make_binary(A, idx_c, str_0):
    for i in range(A.shape[0]):
        if A[i, idx_c] == str_0:
            A[i, idx_c] = 0
        else:
            A[i, idx_c] = 1
    return A


def define_matrix(f_data):
    if f_data == 1:  # Divorce Predictors data set
        data = pd.read_csv('divorce/divorce.csv', delimiter=';', header=None)
        A = data.values[1:, :-1].astype(np.float64)
        f_class = data.values[1:, -1].astype(np.int64)  # 1: divorced, 0: married
        name_attr = np.arange(1, A.shape[1] + 1, 1)
        str_xlabel = 'Attributes'
        str_ylabel = 'Participants'
        fig_size = np.array([20, 15])
    elif f_data == 2:  # e-stat commuting data
        # "Unknown" data have been already removed.
        data = pd.read_csv('estat/1-1.csv', delimiter=',', header=None, dtype=str, encoding='utf-8')
        A = data.values[1:, 1:].astype(np.float64)
        A = np.log(A + 1)
        # A /= (np.tile(np.max(A, axis=1)[:, np.newaxis], (1, A.shape[1])) + 1e-5)
        # data_name = pd.read_csv('estat/name.csv', delimiter=',', header=None, dtype=str, encoding='utf-8')
        name_attr = np.arange(1, A.shape[1] + 1, 1)  # data_name.values[:, 2]
        f_class = np.arange(1, A.shape[0] + 1, 1)  # data_name.values[:, 2]
        str_xlabel = 'Work / school location'
        str_ylabel = 'Home location'
        fig_size = np.array([20, 15])

    return A, f_class, name_attr, str_xlabel, str_ylabel, fig_size

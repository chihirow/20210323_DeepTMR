import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import math, sys, time, dill
import gen_matrix as gm
import plot_results as pr
import torch
from torch import optim
import net
import common as cmn
from matplotlib import cm


def main():
    plt.close('all')
    file_path = './result'
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    dill.load_session('result/compare.pkl')
    S_list = main.S_list
    err = main.err
    err_pca = main.err_pca
    err_svd = main.err_svd
    err_mds = main.err_mds

    pr.plot_compare_mean_std(S_list, err, err_pca, err_svd, err_mds, False)
    pr.plot_compare_mean_std(S_list, err, err_pca, err_svd, err_mds, True)
    pr.plot_compare_scatter(S_list, err, err_pca, err_svd, err_mds, False)
    pr.plot_compare_scatter(S_list, err, err_pca, err_svd, err_mds, True)
    plt.close('all')

    # import pdb;pdb.set_trace()


if __name__ == '__main__':

    main()

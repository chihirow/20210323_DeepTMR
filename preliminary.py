# plot histogram of T when given the true number of clusters (K, H)
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import math, sys, time
import gen_matrix as gm
import plot_results as pr
import torch
from torch import optim
import net
import common as cmn


def main(f_structure, n_gpu):  # f_structure = 1: LBM, 2: Stripe Model, 3: Gradation BM
    start = time.time()
    plt.close('all')
    file_path = './result'
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    np.random.seed(0)
    if n_gpu >= 0:
        torch.manual_seed(0)

    ################################################################################################
    n = 100
    p = n
    f_other_methods = False  # Perform PCA/SVD/MDS
    if f_structure == 1:  # Latent Block Model
        B = np.array([
            [0.9, 0.4, 0.8],
            [0.1, 0.6, 0.2],
            [0.5, 0.3, 0.7]])
        S = 0.05 * np.ones((B.shape[0], B.shape[1]))
        A = gm.gen_matrix(n, p, B, S, 1, 0)  # Gaussian case
    elif f_structure == 2:  # Stripe Model
        B = np.array([0.9, 0.6, 0.3, 0.1])  # Mean of each stripe
        S = 0.05 * np.ones(B.shape[0])  # 0.1
        A = gm.gen_matrix2(n, p, B, S, 1, 0)
    elif f_structure == 3:  # Gradation model
        B = np.array([0.1, 0.9])
        S = 0.05 * np.ones(B.shape[0])
        A = gm.gen_matrix3(n, p, B, S, 1, 0)
    n_epoch = 100
    lr = 1e-2  # Learning rate
    lambda_reg = 1e-10  # Ridge regularization hyperparameter
    n_batch0 = 200
    # -------------------------------------------------------------------------------------------
    A = (A - np.min(A)) / (np.max(A) - np.min(A))  # Make all the entries 0 <= A_ij <= 1
    # ---------------------------------------------------------------------------------------------
    n_units_row = np.array([p, 10, 1])
    n_units_col = np.array([n, 10, 1])
    n_units_out = np.array([2, 10, 1])
    clr_matrix = 'CMRmap_r'
    ################################################################################################

    n_batch = np.min([n_batch0, n * p])  # Batch size
    n_iter = np.int64(np.ceil(np.float64(n_epoch * n * p) / n_batch))  # No. of epochs
    print('n_iter: ' + str(n_iter))
    print('Matrix size: ' + str(n) + ' x ' + str(p))

    A_bar = np.copy(A)
    pr.plot_A(A_bar, r'Matrix $\bar{A}$', 'synthetic' + str(f_structure) + '_input', clr_matrix)
    g1 = np.random.permutation(n)
    g2 = np.random.permutation(p)
    A = A[g1, :][:, g2]  # Random permutation of rows and columns
    pr.plot_A(A, r'Observed matrix $A$', 'synthetic' + str(f_structure) + '_input_permutated', clr_matrix)

    # Define NN model
    model = net.RelationalNet(n_units_row, n_units_col, n_units_out)
    print('No. of params: {}'.format(sum(prm.numel() for prm in model.parameters())) +
          ' (total), {}'.format(sum(prm.numel() for prm in model.parameters() if prm.requires_grad)) + ' (learnable)')
    loss = net.Loss(model)
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    device = torch.device('cuda:{}'.format(n_gpu)) if n_gpu >= 0 else torch.device('cpu')
    torch.cuda.set_device(device)
    model.to(device)
    model.train()

    # NN Training
    loss_all = np.full(n_iter, np.nan)
    cnt = 1
    for t in range(n_iter):
        idx_entry = np.random.choice(n * p, n_batch, replace=False)
        x_row, x_col, y = cmn.define_xy(A, idx_entry, n_batch, device)
        loss_xy = loss.calc_loss(x_row, x_col, y, lambda_reg)
        model.zero_grad()
        loss_xy.backward()
        optimizer.step()
        loss_all[t] = loss_xy.data
        if t / n_iter >= cnt / 30:
            pr.plot_loss(loss_all, 'synthetic' + str(f_structure))  # Plot training loss
            print('>', end='', flush=True)
            cnt += 1
    print('')
    pr.plot_loss(loss_all, 'synthetic' + str(f_structure))  # Plot training loss

    # Plot results
    model.eval()
    A_out, h_row, h_col, order_row, order_col = cmn.calc_result(A, model, device)
    pr.plot_latent(h_row, h_col, 'Row features', 'Column features', 'synthetic' + str(f_structure) + '_features')
    pr.plot_latent(h_row[order_row], h_col[order_col], 'Reordered row features', 'Reordered column features',
                   'synthetic' + str(f_structure) + '_features_sort')

    pr.plot_A(A[order_row, :][:, order_col], 'Reordered input matrix\n(proposed DeepTMR)',
              'synthetic' + str(f_structure) + '_input_sort', clr_matrix)
    pr.plot_A(A_out[order_row, :][:, order_col], 'Reordered output matrix\n(proposed DeepTMR)',
              'synthetic' + str(f_structure) + '_out', clr_matrix)
    elapsed_time = time.time() - start
    print('Overall computation time :{:.2f}'.format(elapsed_time) + '[sec]')

    if f_other_methods:  # Conventional methods
        order_row_pca = cmn.mr_pca_row(A)  # PCA
        order_col_pca = cmn.mr_pca_row(A.T)
        order_row_svd, order_col_svd, A_approx_svd = cmn.mr_svd(A)  # SVD
        order_row_mds = cmn.mr_mds_row(A)  # MDS
        order_col_mds = cmn.mr_mds_row(A.T)
        pr.plot_A(A[order_row_pca, :][:, order_col_pca], 'Reordered input matrix\n(PCA)',
                  'synthetic' + str(f_structure) + '_input_sort_pca', clr_matrix)
        pr.plot_A(A[order_row_svd, :][:, order_col_svd], 'Reordered input matrix\n(SVD)',
                  'synthetic' + str(f_structure) + '_input_sort_svd', clr_matrix)
        pr.plot_A(A_approx_svd[order_row_svd, :][:, order_col_svd], 'Reordered output matrix\n(SVD)',
                  'synthetic' + str(f_structure) + '_out_svd', clr_matrix)
        pr.plot_A(A[order_row_mds, :][:, order_col_mds], 'Reordered input matrix\n(MDS)',
                  'synthetic' + str(f_structure) + '_input_sort_mds', clr_matrix)

        # if f_structure == 2:  # Compute squared error
        #     i_order, j_order, err0 = cmn.select_order(order_row, order_col, A_bar, A)
        #     i_order_pca, j_order_pca, err0_pca = cmn.select_order(order_row_pca, order_col_pca, A_bar, A)
        #     i_order_svd, j_order_svd, err0_svd = cmn.select_order(order_row_svd, order_col_svd, A_bar, A)
        #     i_order_mds, j_order_mds, err0_mds = cmn.select_order(order_row_mds, order_col_mds, A_bar, A)
        #     print('Squared error, DeepTMR: {:.5f}'.format(err0))
        #     print('Squared error, PCA    : {:.5f}'.format(err0_pca))
        #     print('Squared error, SVD    : {:.5f}'.format(err0_svd))
        #     print('Squared error, MDS    : {:.5f}'.format(err0_mds))

    # import pdb;pdb.set_trace()


if __name__ == '__main__':

    main(int(sys.argv[1]), int(sys.argv[2]))

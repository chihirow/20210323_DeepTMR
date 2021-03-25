import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import math, sys, time, dill, copy
import gen_matrix as gm
import plot_results as pr
import torch
from torch import optim
import net
import common as cmn
from matplotlib import cm


def plot_A_sub(A, str_title, str_file, clr_matrix, n_row_fig, n_col_fig, idx_fig, fig_number):
    n = A.shape[0]
    p = A.shape[1]
    plt.figure(fig_number)
    plt.subplot(n_row_fig, n_col_fig, idx_fig)
    plt.imshow(A, cmap=clr_matrix, vmin=0, vmax=1, interpolation='none')
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.colorbar(ticks=np.arange(0, 1.1, 0.2))
    plt.xlim([-0.5, p - 0.5])
    plt.ylim([-0.5, n - 0.5])
    plt.title(str_title)
    plt.gca().invert_yaxis()
    plt.savefig('result/A_' + str_file + '.png', bbox_inches='tight')
    plt.savefig('result/A_' + str_file + '.eps', bbox_inches='tight')
    # plt.close()


def main(n_gpu):
    start = time.time()
    plt.close('all')
    file_path = './result'
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    np.random.seed(0)
    if n_gpu >= 0:
        torch.manual_seed(0)

    ################################################################################################
    n_replicate = 10
    n_replicate_deeptmr = 5  # No. of trials for trainng DNN with the same data matrix
    n_iter_last = 100  # For each data, choose the model with minimum mean error in the last n_iter_last iterations.
    n = 100
    p = n
    B = np.array([0.9, 0.1])
    S_list = np.arange(0.03, 0.31, 0.03)
    lr = 1e-2
    n_epoch = 200
    lambda_reg = 1e-10  # Ridge regularization hyperparameter
    n_batch0 = 200
    # -------------------------------------------------------------------------------------------
    n_units_row = np.array([p, 10, 1])
    n_units_col = np.array([n, 10, 1])
    n_units_out = np.array([2, 10, 1])
    clr_matrix = 'CMRmap_r'
    # ---------------------------------------------------------------------------------------------
    n_batch = np.min([n_batch0, n * p])  # Batch size
    n_iter = np.int64(np.ceil(np.float64(n_epoch * n * p) / n_batch))  # No. of epochs
    print('n_iter: ' + str(n_iter))
    print('Matrix size: ' + str(n) + ' x ' + str(p))
    ################################################################################################

    err = np.zeros((S_list.shape[0], n_replicate))
    err_pca = np.zeros((S_list.shape[0], n_replicate))
    err_svd = np.zeros((S_list.shape[0], n_replicate))
    err_mds = np.zeros((S_list.shape[0], n_replicate))

    n_fig = S_list.shape[0]
    n_col_fig = 5
    n_row_fig = n_fig // n_col_fig
    fig_size_A = [6.5 * n_col_fig, 6 * n_row_fig]
    plt.rcParams["font.size"] = 18
    fig_A_bar = plt.figure(figsize=(fig_size_A[0], fig_size_A[1]))
    fig_A = plt.figure(figsize=(fig_size_A[0], fig_size_A[1]))
    fig_A_deeptmr = plt.figure(figsize=(fig_size_A[0], fig_size_A[1]))
    fig_A_out_deeptmr = plt.figure(figsize=(fig_size_A[0], fig_size_A[1]))
    fig_A_pca = plt.figure(figsize=(fig_size_A[0], fig_size_A[1]))
    fig_A_svd = plt.figure(figsize=(fig_size_A[0], fig_size_A[1]))
    fig_A_mds = plt.figure(figsize=(fig_size_A[0], fig_size_A[1]))

    for k in range(S_list.shape[0]):
        print('[sigma = {:.2f}'.format(S_list[k]) + ']')
        for m in range(n_replicate):
            A_bar, P_bar = gm.gen_matrix4(n, p, B, S_list[k], 1, n_replicate * k + m + 1)
            A_bar = (A_bar - np.min(A_bar)) / (np.max(A_bar) - np.min(A_bar))  # Make all the entries 0 <= A_ij <= 1
            g1 = np.random.permutation(n)
            g2 = np.random.permutation(p)
            A = np.copy(A_bar)
            A = A[g1, :][:, g2]  # Random permutation of rows and columns
            P = np.copy(P_bar)
            P = P[g1, :][:, g2]

            # Define NN model
            for i_tr in range(n_replicate_deeptmr):
                model = net.RelationalNet(n_units_row, n_units_col, n_units_out)
                loss = net.Loss(model)
                optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
                device = torch.device('cuda:{}'.format(n_gpu)) if n_gpu >= 0 else torch.device('cpu')
                torch.cuda.set_device(device)
                model.to(device)
                model.train()

                # NN Training
                loss_all = np.full(n_iter, np.nan)
                for t in range(n_iter):
                    idx_entry = np.random.choice(n * p, n_batch, replace=False)
                    x_row, x_col, y = cmn.define_xy(A, idx_entry, n_batch, device)
                    loss_xy = loss.calc_loss(x_row, x_col, y, lambda_reg)
                    model.zero_grad()
                    loss_xy.backward()
                    optimizer.step()
                    loss_all[t] = loss_xy.data
                if i_tr == 0:  # Choose the best model (minimum reconstruction error at the last iteration)
                    model_opt = copy.deepcopy(model)
                    loss_all_opt = np.copy(loss_all)
                else:
                    if np.mean(loss_all[-n_iter_last:]) < np.mean(loss_all_opt[-n_iter_last:]):
                        model_opt = copy.deepcopy(model)
                        loss_all_opt = np.copy(loss_all)
            model = copy.deepcopy(model_opt)
            loss_all = np.copy(loss_all_opt)

            # Plot results
            model.eval()
            A_out, h_row, h_col, order_row, order_col = cmn.calc_result(A, model, device)

            order_row_pca = cmn.mr_pca_row(A)  # PCA
            order_col_pca = cmn.mr_pca_row(A.T)
            order_row_svd, order_col_svd, A_approx_svd = cmn.mr_svd(A)  # SVD
            order_row_mds = cmn.mr_mds_row(A)  # MDS
            order_col_mds = cmn.mr_mds_row(A.T)

            # Compute squared error
            order_row_opt, order_col_opt, err[k, m] = \
                cmn.select_order(order_row, order_col, P, P_bar)
            order_row_opt_pca, order_col_opt_pca, err_pca[k, m] = \
                cmn.select_order(order_row_pca, order_col_pca, P, P_bar)
            order_row_opt_svd, order_col_opt_svd, err_svd[k, m] = \
                cmn.select_order(order_row_svd, order_col_svd, P, P_bar)
            order_row_opt_mds, order_col_opt_mds, err_mds[k, m] = \
                cmn.select_order(order_row_mds, order_col_mds, P, P_bar)

            if m == 0:
                pr.plot_loss(loss_all, 'synthetic4_' + str(k + 1))  # Plot training loss
                plot_A_sub(A_bar, r'Matrix $\bar{A}$,' + '\n' + r'$t=' + str(k + 1) + r'$',
                           'synthetic4_input', clr_matrix, n_row_fig, n_col_fig, k + 1, fig_A_bar.number)
                plot_A_sub(A, r'Observed matrix $A$,' + '\n' + r'$t=' + str(k + 1) + r'$',
                           'synthetic4_input_permutated', clr_matrix, n_row_fig, n_col_fig, k + 1, fig_A.number)
                plot_A_sub(A[order_row_opt, :][:, order_col_opt], 'Reordered input matrix,\n' +
                           r'$t=' + str(k + 1) + r'$' + ' (DeepTMR)',
                           'synthetic4_input_sort', clr_matrix, n_row_fig, n_col_fig, k + 1, fig_A_deeptmr.number)
                plot_A_sub(A_out[order_row_opt, :][:, order_col_opt], 'Reordered output\nmatrix, ' +
                           r'$t=' + str(k + 1) + r'$' + '\n(DeepTMR)',
                           'synthetic4_out', clr_matrix, n_row_fig, n_col_fig, k + 1, fig_A_out_deeptmr.number)
                plot_A_sub(A[order_row_opt_pca, :][:, order_col_opt_pca], 'Reordered input matrix,\n' +
                           r'$t=' + str(k + 1) + r'$' + ' (SVD-Angle)',
                           'synthetic4_input_sort_pca', clr_matrix, n_row_fig, n_col_fig, k + 1, fig_A_pca.number)
                plot_A_sub(A[order_row_opt_svd, :][:, order_col_opt_svd], 'Reordered input matrix,\n' +
                           r'$t=' + str(k + 1) + r'$' + ' (SVD-Rank-One)',
                           'synthetic4_input_sort_svd', clr_matrix, n_row_fig, n_col_fig, k + 1, fig_A_svd.number)
                plot_A_sub(A[order_row_opt_mds, :][:, order_col_opt_mds], 'Reordered input matrix,\n' +
                           r'$t=' + str(k + 1) + r'$' + ' (MDS)',
                           'synthetic4_input_sort_mds', clr_matrix, n_row_fig, n_col_fig, k + 1, fig_A_mds.number)
            print('>', end='', flush=True)
        print('')
        print(' Error (DeepTMR)     : {:.2g}'.format(np.mean(err[k])) + ' +- {:.2g}'.format(np.std(err[k])) +
              ', [{:.2g}'.format(np.min(err[k])) + ', {:.2g}]'.format(np.max(err[k])))
        print(' Error (SVD-Angle)   : {:.2g}'.format(np.mean(err_pca[k])) + ' +- {:.2g}'.format(np.std(err_pca[k])) +
              ', [{:.2g}'.format(np.min(err_pca[k])) + ', {:.2g}]'.format(np.max(err_pca[k])))
        print(' Error (SVD-Rank-One): {:.2g}'.format(np.mean(err_svd[k])) + ' +- {:.2g}'.format(np.std(err_svd[k])) +
              ', [{:.2g}'.format(np.min(err_svd[k])) + ', {:.2g}]'.format(np.max(err_svd[k])))
        print(' Error (MDS)         : {:.2g}'.format(np.mean(err_mds[k])) + ' +- {:.2g}'.format(np.std(err_mds[k])) +
              ', [{:.2g}'.format(np.min(err_mds[k])) + ', {:.2g}]'.format(np.max(err_mds[k])))
        elapsed_time = time.time() - start
        print('Overall computation time :{:.2f}'.format(elapsed_time) + '[sec]')

        pr.plot_compare_mean_std(S_list, err, err_pca, err_svd, err_mds, False)
        pr.plot_compare_mean_std(S_list, err, err_pca, err_svd, err_mds, True)
        pr.plot_compare_scatter(S_list, err, err_pca, err_svd, err_mds, False)
        pr.plot_compare_scatter(S_list, err, err_pca, err_svd, err_mds, True)
    plt.close('all')

    main.S_list = S_list
    main.err = err
    main.err_pca = err_pca
    main.err_svd = err_svd
    main.err_mds = err_mds
    dill.dump_session('result/compare.pkl')

    # import pdb;pdb.set_trace()


if __name__ == '__main__':

    main(int(sys.argv[1]))

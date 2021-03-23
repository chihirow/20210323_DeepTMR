import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys, time
import numpy as np
import matplotlib.pyplot as plt
import plot_results as pr
import torch
from torch import optim
import net
import practical_gen_matrix as pgm
import common as cmn


def main(f_data, n_gpu):
    start = time.time()
    plt.close('all')
    file_path = './result'
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    np.random.seed(0)
    if n_gpu >= 0:
        torch.manual_seed(0)

    ################################################################################################
    f_other_methods = False  # Perform PCA/SVD/MDS
    color_y1 = 'deepskyblue'  # Color for class 1
    color_y0 = 'gold'  # Color for class 0
    if f_data == 1:  # Divorce Predictors data set
        # https://archive.ics.uci.edu/ml/datasets/Divorce+Predictors+data+set
        str_class = ['Married', 'Divorced']
        n_epoch = 200
        pr.plot_A_ylabel(color_y1, color_y0, str_class, 'legend_practical' + str(f_data))
    elif f_data == 2:  # e-stat commuting data
        # https://www.e-stat.go.jp/stat-search/files?page=1&query=%E8%A1%8C%E6%94%BF%E5%8C%BA%E7%94%BB%E9%96%93%E7%A7%BB%E5%8B%95%E4%BA%BA%E5%93%A1%E8%A1%A8&layout=dataset&stat_infid=000031598030
        n_epoch = 100
    # ---------------------------------------------------------------------------------------------
    lr = 1e-2
    lambda_reg = 1e-10
    n_batch0 = 500
    A, f_class, name_attr, str_xlabel, str_ylabel, fig_size = pgm.define_matrix(f_data)
    A = (A - np.min(A)) / (np.max(A) - np.min(A))  # Make all the entries 0 <= A_ij <= 1
    n = A.shape[0]
    p = A.shape[1]
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

    pr.plot_A_practical(A, str_xlabel, str_ylabel, fig_size, clr_matrix, 'practical' + str(f_data) + '_input',
                        r'Observed matrix $A$', f_class, name_attr, color_y1, color_y0, f_data)

    # Define NN model
    model = net.RelationalNet(n_units_row, n_units_col, n_units_out)
    print('No. of params: {}'.format(sum(prm.numel() for prm in model.parameters())) +
          ' (total), {}'.format(sum(prm.numel() for prm in model.parameters() if prm.requires_grad)) + ' (learnable)')
    loss = net.Loss(model)
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
            pr.plot_loss(loss_all, 'practical' + str(f_data))  # Plot training loss
            print('>', end='', flush=True)
            cnt += 1
    print('')
    pr.plot_loss(loss_all, 'practical' + str(f_data))  # Plot training loss

    # Plot results
    model.eval()
    A_out, h_row, h_col, order_row, order_col = cmn.calc_result(A, model, device)
    pr.plot_latent(h_row, h_col, 'Row features', 'Column features', 'practical' + str(f_data) + '_features')
    pr.plot_latent(h_row[order_row], h_col[order_col], 'Reordered row features', 'Reordered column features',
                   'practical' + str(f_data) + '_features_sort')
    pr.plot_A_practical(A[order_row, :][:, order_col], str_xlabel, str_ylabel, fig_size, clr_matrix,
                        'practical' + str(f_data) + '_input_sort', 'Reordered input matrix\n(proposed DeepTMR)',
                        f_class[order_row], name_attr[order_col], color_y1, color_y0, f_data)
    pr.plot_A_practical(A_out[order_row, :][:, order_col], str_xlabel, str_ylabel, fig_size, clr_matrix,
                        'practical' + str(f_data) + '_out', 'Reordered output matrix\n(proposed DeepTMR)',
                        f_class[order_row], name_attr[order_col], color_y1, color_y0, f_data)
    elapsed_time = time.time() - start
    print('Overall computation time :{:.2f}'.format(elapsed_time) + '[sec]')

    if f_other_methods:  # Conventional methods
        order_row_pca = cmn.mr_pca_row(A)  # PCA
        order_col_pca = cmn.mr_pca_row(A.T)
        order_row_svd, order_col_svd, A_approx_svd = cmn.mr_svd(A)  # SVD
        order_row_mds = cmn.mr_mds_row(A)  # MDS
        order_col_mds = cmn.mr_mds_row(A.T)
        pr.plot_A_practical(A[order_row_pca, :][:, order_col_pca], str_xlabel, str_ylabel, fig_size, clr_matrix,
                            'practical' + str(f_data) + '_input_sort_pca', 'Reordered input matrix\n(PCA)',
                            f_class[order_row_pca], name_attr[order_col_pca], color_y1, color_y0, f_data)
        pr.plot_A_practical(A[order_row_svd, :][:, order_col_svd], str_xlabel, str_ylabel, fig_size, clr_matrix,
                            'practical' + str(f_data) + '_input_sort_svd', 'Reordered input matrix\n(SVD)',
                            f_class[order_row_svd], name_attr[order_col_svd], color_y1, color_y0, f_data)
        pr.plot_A_practical(A_approx_svd[order_row_svd, :][:, order_col_svd], str_xlabel, str_ylabel, fig_size, clr_matrix,
                            'practical' + str(f_data) + '_out_svd', 'Reordered output matrix\n(SVD)',
                            f_class[order_row_svd], name_attr[order_col_svd], color_y1, color_y0, f_data)
        pr.plot_A_practical(A[order_row_mds, :][:, order_col_mds], str_xlabel, str_ylabel, fig_size, clr_matrix,
                            'practical' + str(f_data) + '_input_sort_mds', 'Reordered input matrix\n(MDS)',
                            f_class[order_row_mds], name_attr[order_col_mds], color_y1, color_y0, f_data)

    # import pdb;pdb.set_trace()


if __name__ == '__main__':

    main(int(sys.argv[1]), int(sys.argv[2]))

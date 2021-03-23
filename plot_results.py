import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.special as spys
import math


def plot_loss(loss_all, str_file):
    plt.rcParams['font.size'] = 25
    plt.figure(figsize=(12, 4))
    x = np.arange(0, loss_all.shape[0], 1)
    plt.plot(x, loss_all, color=np.zeros(3), label='Overall training loss')
    plt.tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False)
    plt.tick_params(bottom=True, left=True, right=False, top=False)
    plt.xlabel('Iterations')
    plt.ylabel('Training loss')
    plt.savefig('result/loss_' + str_file + '.png', bbox_inches='tight')
    plt.close()


def plot_latent(h_row, h_col, str_title_row, str_title_col, str_file):
    ylim_margin = 0.1
    n = h_row.shape[0]
    p = h_col.shape[0]
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(1, n + 1, 1), h_row, color='k', ls='None', marker='.')
    plt.xlabel('Rows')
    plt.title(str_title_row)
    plt.ylim([-ylim_margin, 1 + ylim_margin])
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(1, p + 1, 1), h_col, color='k', ls='None', marker='.')
    plt.xlabel('Columns')
    plt.title(str_title_col)
    plt.ylim([-ylim_margin, 1 + ylim_margin])
    plt.tight_layout()
    plt.savefig('result/' + str_file + '.png', bbox_inches='tight')
    plt.savefig('result/' + str_file + '.eps', bbox_inches='tight')
    plt.close()


def plot_A(A, str_title, str_file, clr_matrix):
    n = A.shape[0]
    p = A.shape[1]
    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(6, 5))
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
    plt.close()


def plot_A_ylabel(color_y1, color_y0, str_class, str_file):
    plt.rcParams["font.size"] = 25
    plt.figure(figsize=(6, 12))
    p_height = 0.2
    p_width = 0.2
    plt.text(0, p_height, r'$\bullet$', color=color_y1, verticalalignment='center', fontsize=50)
    plt.text(p_width, p_height, str_class[1], verticalalignment='center')
    plt.text(0, -p_height, r'$\bullet$', color=color_y0, verticalalignment='center', fontsize=50)
    plt.text(p_width, -p_height, str_class[0], verticalalignment='center')
    plt.xlim([-0.2, 1])
    plt.ylim(15 * p_height * np.array([-1, 1]))
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.axis('off')
    plt.savefig('result/' + str_file + '.png', bbox_inches='tight')
    plt.savefig('result/' + str_file + '.eps', bbox_inches='tight')
    plt.close()


def plot_A_practical(A, str_xlabel, str_ylabel, fig_size, clr_matrix, str_file, str_title, f_class, name_attr,
                     color_y1, color_y0, f_data):
    plt.rcParams["font.size"] = 40
    plt.figure(figsize=(fig_size[0], fig_size[1]))
    plt.imshow(A, cmap=clr_matrix, aspect='auto', interpolation='none', vmin=0, vmax=1)
    plt.colorbar(ticks=np.arange(0, 1.1, 0.2))
    plt.title(str_title)
    plt.xlim([-0.5, A.shape[1] - 0.5])
    plt.ylim([-0.5, A.shape[0] - 0.5])

    if f_data == 1:  # Divorce Predictors data set
        plt.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
        plt.xticks(np.arange(0, A.shape[1], 1), name_attr, rotation=90, fontsize=18)  # 18
        for i in range(A.shape[0]):
            if f_class[i] == 1:
                clr = color_y1
            else:
                clr = color_y0
            plt.plot([-0.5, -0.5], [i - 0.5, i + 0.5], color=clr, lw=10)
    elif f_data == 2:  # e-stat commuting data
        plt.tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False)
        width_mesh = 10
        x_minor = np.arange(-0.5, A.shape[1], width_mesh)
        y_minor = np.arange(-0.5, A.shape[0], width_mesh)
        xtick_name = np.zeros(x_minor.shape[0], dtype=object)
        ytick_name = np.zeros(y_minor.shape[0], dtype=object)
        for i in range(xtick_name.shape[0]):
            xtick_name[i] = '(C' + str(i + 1) + ')'
        for i in range(ytick_name.shape[0]):
            ytick_name[i] = '(R' + str(i + 1) + ')'
        ax = plt.gca()
        ax.set_xticks(x_minor, minor=True)
        ax.set_yticks(y_minor, minor=True)
        ax.grid(which='minor', color='c', linestyle='-', linewidth=1)
        plt.xticks(x_minor + (width_mesh / 2), xtick_name, rotation=90, fontsize=24)
        plt.yticks(y_minor + (width_mesh / 2), ytick_name, rotation=0, fontsize=24)

    plt.tick_params(bottom=False, left=False, right=False, top=False)
    plt.xlabel(str_xlabel, fontsize=40)
    plt.ylabel(str_ylabel, fontsize=40)
    plt.gca().invert_yaxis()
    plt.savefig('result/' + str_file + '.png', bbox_inches='tight')
    plt.savefig('result/' + str_file + '.eps', bbox_inches='tight')
    plt.close()

    if f_data == 2:  # e-stat commuting data
        p_height = 0.2
        plt.rcParams["font.size"] = 24
        plt.figure(figsize=(fig_size[0] // 2, fig_size[1]))
        plt.text(0, 0, 'From left to right in each section,', va='center', ha='left')
        cnt = 0
        for i in range(xtick_name.shape[0]):  # (C1), (C2), ...
            str_i = xtick_name[i]
            for j in range(width_mesh):
                if cnt < A.shape[1]:
                    if j == 0:
                        str_i = str_i + ' ' + str(name_attr[cnt])
                    else:
                        str_i = str_i + ', ' + str(name_attr[cnt])
                cnt += 1
            plt.text(0, -(i + 1) * p_height, str_i, va='center', ha='left')
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.tick_params(bottom=False, left=False, right=False, top=False)
        plt.ylim([-p_height * (xtick_name.shape[0] + 1), p_height])
        plt.axis('off')
        plt.savefig('result/' + str_file + '_Clist.png', bbox_inches='tight')
        plt.savefig('result/' + str_file + '_Clist.eps', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(fig_size[0] // 2, fig_size[1]))
        plt.text(0, 0, 'From top to bottom in each section,', va='center', ha='left')
        cnt = 0
        for i in range(ytick_name.shape[0]):  # (R1), (R2), ...
            str_i = ytick_name[i]
            for j in range(width_mesh):
                if cnt < A.shape[0]:
                    if j == 0:
                        str_i = str_i + ' ' + str(f_class[cnt])
                    else:
                        str_i = str_i + ', ' + str(f_class[cnt])
                cnt += 1
            plt.text(0, -(i + 1) * p_height, str_i, va='center', ha='left')
        plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
        plt.tick_params(bottom=False, left=False, right=False, top=False)
        plt.ylim([-p_height * (ytick_name.shape[0] + 1), p_height])
        plt.axis('off')
        plt.savefig('result/' + str_file + '_Rlist.png', bbox_inches='tight')
        plt.savefig('result/' + str_file + '_Rlist.eps', bbox_inches='tight')
        plt.close()

def write_cities():
    n_row = 63
    data_name = pd.read_csv('estat/name.csv', delimiter=',', header=None, dtype=str, encoding='utf-8')
    name_attr = data_name.values[:, 1]
    n_col = np.ceil(name_attr.shape[0] / n_row).astype(np.int64)
    f = open('result/cities.txt', 'w')
    for i in range(name_attr.shape[0]):
        f.writelines(str_i + '\n')
    f.close()


def plot_compare_mean_std(S_list, err, err_pca, err_svd, err_mds, f_ylog):
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(10, 5))
    plt.errorbar(S_list, np.mean(err, axis=1), yerr=np.std(err, axis=1), ls='-', lw=2, capsize=5, fmt='.',
                 color=cm.plasma(0 / 4), label='DeepTMR')
    plt.errorbar(S_list, np.mean(err_pca, axis=1), yerr=np.std(err_pca, axis=1), ls='-', lw=2, capsize=5, fmt='.',
                 color=cm.plasma(1 / 4), label='PCA')
    plt.errorbar(S_list, np.mean(err_svd, axis=1), yerr=np.std(err_svd, axis=1), ls='-', lw=2, capsize=5, fmt='.',
                 color=cm.plasma(2 / 4), label='SVD')
    plt.errorbar(S_list, np.mean(err_mds, axis=1), yerr=np.std(err_mds, axis=1), ls='-', lw=2, capsize=5, fmt='.',
                 color=cm.plasma(3 / 4), label='MDS')
    if f_ylog:
        plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.xlabel(r'$\sigma$')
    plt.title('Matrix reordering error')
    plt.tight_layout()
    if f_ylog:
        plt.savefig('result/synthetic4_compare_mean_ylog.png', bbox_inches='tight')
        plt.savefig('result/synthetic4_compare_mean_ylog.eps', bbox_inches='tight')
    else:
        plt.savefig('result/synthetic4_compare_mean.png', bbox_inches='tight')
        plt.savefig('result/synthetic4_compare_mean.eps', bbox_inches='tight')
    plt.close()


def plot_compare_scatter(S_list, err, err_pca, err_svd, err_mds, f_ylog):
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(12, 8))
    err2 = np.copy(err)
    err2[err2 == 0] = np.inf
    err2_pca = np.copy(err_pca)
    err2_pca[err2_pca == 0] = np.inf
    err2_svd = np.copy(err_svd)
    err2_svd[err2_svd == 0] = np.inf
    err2_mds = np.copy(err_mds)
    err2_mds[err2_mds == 0] = np.inf
    ylim0 = np.min([np.min(err2), np.min(err2_pca), np.min(err2_svd), np.min(err2_mds)]) * 0.9
    ylim1 = np.max([np.max(err), np.max(err_pca), np.max(err_svd), np.max(err_mds)]) * 1.1
    n_S = err.shape[0]
    n_replicate = err.shape[1]
    x_scatter = np.zeros(n_S * n_replicate)
    err_scatter = np.zeros(n_S * n_replicate)
    err_scatter_pca = np.zeros(n_S * n_replicate)
    err_scatter_svd = np.zeros(n_S * n_replicate)
    err_scatter_mds = np.zeros(n_S * n_replicate)
    for k in range(n_S):
        for m in range(n_replicate):
            x_scatter[n_replicate * k + m] = S_list[k]
            err_scatter[n_replicate * k + m] = err[k, m]
            err_scatter_pca[n_replicate * k + m] = err_pca[k, m]
            err_scatter_svd[n_replicate * k + m] = err_svd[k, m]
            err_scatter_mds[n_replicate * k + m] = err_mds[k, m]

    plt.subplot(1, 4, 1)
    plt.scatter(x_scatter, err_scatter, marker='.', color='k')
    plt.xlabel(r'$\sigma$')
    plt.title('Matrix reordering\nerror (DeepTMR)')
    if f_ylog:
        plt.yscale('log')
    plt.ylim([ylim0, ylim1])
    plt.grid(axis='y')
    #
    plt.subplot(1, 4, 2)
    plt.scatter(x_scatter, err_scatter_pca, marker='.', color='k')
    plt.xlabel(r'$\sigma$')
    plt.title('Matrix reordering\nerror (PCA)')
    if f_ylog:
        plt.yscale('log')
    plt.ylim([ylim0, ylim1])
    plt.grid(axis='y')
    #
    plt.subplot(1, 4, 3)
    plt.scatter(x_scatter, err_scatter_svd, marker='.', color='k')
    plt.xlabel(r'$\sigma$')
    plt.title('Matrix reordering\nerror (SVD)')
    if f_ylog:
        plt.yscale('log')
    plt.ylim([ylim0, ylim1])
    plt.grid(axis='y')
    #
    plt.subplot(1, 4, 4)
    plt.scatter(x_scatter, err_scatter_mds, marker='.', color='k')
    plt.xlabel(r'$\sigma$')
    plt.title('Matrix reordering\nerror (MDS)')
    if f_ylog:
        plt.yscale('log')
    plt.ylim([ylim0, ylim1])
    plt.grid(axis='y')

    plt.tight_layout()
    if f_ylog:
        plt.savefig('result/synthetic4_compare_ylog.png', bbox_inches='tight')
        plt.savefig('result/synthetic4_compare_ylog.eps', bbox_inches='tight')
    else:
        plt.savefig('result/synthetic4_compare.png', bbox_inches='tight')
        plt.savefig('result/synthetic4_compare.eps', bbox_inches='tight')
    plt.close()

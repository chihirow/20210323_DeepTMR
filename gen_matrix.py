import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit('f8[:, :](i8, i8, f8[:, :], f8[:, :], i8, i8)', nopython=True)
def gen_matrix(n, p, B, S, f_dist, n_seed):
    np.random.seed(n_seed)
    A = np.zeros((n, p))
    s_block_row = np.int64(np.ceil(np.float64(n) / B.shape[0]))
    s_block_col = np.int64(np.ceil(np.float64(p) / B.shape[1]))
    for k in range(B.shape[0]):
        i_k1 = k * s_block_row
        i_k2 = min([(k + 1) * s_block_row, n])
        for h in range(B.shape[1]):
            j_h1 = h * s_block_col
            j_h2 = min([(h + 1) * s_block_col, p])
            if f_dist == 1:  # normal
                A[i_k1:i_k2, j_h1:j_h2] = np.random.normal(B[k, h], S[k, h], (i_k2 - i_k1, j_h2 - j_h1))
            elif f_dist == 2:  # Bernoulli
                A[i_k1:i_k2, j_h1:j_h2] = np.random.binomial(1, B[k, h], (i_k2 - i_k1, j_h2 - j_h1))
            elif f_dist == 3:  # Poisson
                A[i_k1:i_k2, j_h1:j_h2] = np.random.poisson(B[k, h], (i_k2 - i_k1, j_h2 - j_h1))

    return A


@jit('f8[:, :](i8, i8, f8[:], f8[:], i8, i8)', nopython=True)
def gen_matrix2(n, p, B, S, f_dist, n_seed):  # Stripe Model
    np.random.seed(n_seed)
    K = B.shape[0]
    s_block = np.int64(np.ceil(np.float64(n + p) / K))  # (n + p) // K
    A = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            k = np.int64(np.floor((i + j) / s_block))
            P_ij = B[k]
            sigma_ij = S[k]
            if f_dist == 1:  # normal
                A[i, j] = np.random.normal(P_ij, sigma_ij)
            elif f_dist == 2:  # Bernoulli
                A[i, j] = np.random.binomial(1, P_ij)
            elif f_dist == 3:  # Poisson
                A[i, j] = np.random.poisson(P_ij)

    return A


@jit('f8[:, :](i8, i8, f8[:], f8[:], i8, i8)', nopython=True)
def gen_matrix3(n, p, B, S, f_dist, n_seed):  # Gradation model
    np.random.seed(n_seed)
    s_block_row = np.int64(np.ceil(np.float64(n) / 2))  # n // 2
    s_block_col = np.int64(np.ceil(np.float64(p) / 2))  # p // 2
    A = np.zeros((n, p))
    for i in range(n):
        k = np.int64(np.floor(i / s_block_row))  # i // s_block_row  # 0 or 1
        for j in range(p):
            h = np.int64(np.floor(j / s_block_col))  # j // s_block_col  # 0 or 1
            if (k == 0) and (h == 0):
                P_ij = (B[1] - B[0]) * j / (s_block_col - 1) + B[0]
                sigma_ij = (S[1] - S[0]) * j / (s_block_col - 1) + S[0]
            elif (k == 0) and (h == 1):
                # P_ij = (B[1] - B[0]) * i / (s_block_row - 1) + B[0]
                # sigma_ij = (S[1] - S[0]) * i / (s_block_row - 1) + S[0]
                P_ij = B[0]
                sigma_ij = S[0]
            elif (k == 1) and (h == 0):
                # P_ij = (B[1] - B[0]) * (i - s_block_row) / (s_block_row - 1) + B[0]
                # sigma_ij = (S[1] - S[0]) * (i - s_block_row) / (s_block_row - 1) + S[0]
                P_ij = B[0]
                sigma_ij = S[0]
            elif (k == 1) and (h == 1):
                # P_ij = (B[1] - B[0]) * (j - s_block_col) / (s_block_col - 1) + B[0]
                # sigma_ij = (S[1] - S[0]) * (j - s_block_col) / (s_block_col - 1) + S[0]
                P_ij = B[0]
                sigma_ij = S[0]
            if f_dist == 1:  # normal
                A[i, j] = np.random.normal(P_ij, sigma_ij)
            elif f_dist == 2:  # Bernoulli
                A[i, j] = np.random.binomial(1, P_ij)
            elif f_dist == 3:  # Poisson
                A[i, j] = np.random.poisson(P_ij)

    return A


@jit('Tuple((f8[:, :], f8[:, :]))(i8, i8, f8[:], f8, i8, i8)', nopython=True)
def gen_matrix4(n, p, B, sigma_ij, f_dist, n_seed):  # Oblique gradation model
    np.random.seed(n_seed)
    A = np.zeros((n, p))
    P = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            P[i, j] = B[0] - (B[0] - B[1]) * np.abs(i - j) / max([n, p])
            if f_dist == 1:  # normal
                A[i, j] = np.random.normal(P[i, j], sigma_ij)
            elif f_dist == 2:  # Bernoulli
                A[i, j] = np.random.binomial(1, P[i, j])
            elif f_dist == 3:  # Poisson
                A[i, j] = np.random.poisson(P[i, j])

    return A, P


def plot_g(A, g1, g2, str_title, str_file):  # plot cluster assignment
    n = A.shape[0]
    p = A.shape[1]
    order1 = np.argsort(g1)  # ascending order of rows
    order2 = np.argsort(g2)  # ascending order of columns

    A_sort = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            A_sort[i, j] = A[order1[i], order2[j]]

    plt.rcParams["font.size"] = 18
    plt.figure(figsize=(5, 5))
    plt.imshow(A_sort, cmap='binary', vmin=0, vmax=1)
    # plt.imshow(A_sort, cmap='binary', vmin=0, vmax=1)
    nc = 0
    for c in range(np.max(g1)):
        nc = nc + np.sum(g1 == c)
        plt.plot([-0.5, p - 0.5], [nc - 0.5, nc - 0.5], 'b-', lw=2)
    nc = 0
    for c in range(np.max(g2)):
        nc = nc + np.sum(g2 == c)
        plt.plot([nc - 0.5, nc - 0.5], [-0.5, n - 0.5], 'b-', lw=2)
    plt.tick_params(labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
    plt.tick_params(bottom=False,
                    left=False,
                    right=False,
                    top=False)
    plt.colorbar()
    # plt.colorbar(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xlim([-0.5, p - 0.5])
    plt.ylim([-0.5, n - 0.5])
    plt.title(str_title)
    plt.gca().invert_yaxis()
    plt.savefig('result/A_' + str_file + '.png', bbox_inches='tight')
    plt.savefig('result/A_' + str_file + '.eps', bbox_inches='tight')
    plt.close()


@jit
def plot_g_practical(A, g1, g2, name, str_title, str_file):  # plot cluster assignment
    n = A.shape[0]
    p = A.shape[1]
    order1 = np.argsort(g1)  # ascending order of rows
    order2 = np.argsort(g2)  # ascending order of columns

    A_sort = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            A_sort[i, j] = A[order1[i], order2[j]]

    plt.rcParams["font.size"] = 70
    plt.figure(figsize=(70, 20))
    plt.imshow(A_sort, cmap='binary', aspect=0.25)
    nc = 0
    for c in range(np.max(g1)):
        nc = nc + np.sum(g1 == c)
        plt.plot([-0.5, p - 0.5], [nc - 0.5, nc - 0.5], 'b-', lw=3)
    nc = 0
    for c in range(np.max(g2)):
        nc = nc + np.sum(g2 == c)
        plt.plot([nc - 0.5, nc - 0.5], [-0.5, n - 0.5], 'b-', lw=3)
    plt.tick_params(labelbottom=True,
                    labelleft=False,
                    labelright=False,
                    labeltop=False)
    plt.tick_params(bottom=True,
                    left=False,
                    right=False,
                    top=False)
    plt.colorbar()
    plt.xlim([-0.5, p - 0.5])
    plt.ylim([-0.5, n - 0.5])
    plt.xlabel('Words')
    plt.ylabel('Papers')
    plt.xticks(np.arange(len(name)), name, fontsize=40, rotation=90)
    # plt.title(str_title)
    plt.gca().invert_yaxis()
    plt.savefig('result/A_' + str_file + '.png', bbox_inches='tight')
    plt.close()


@jit
def sort_g(g1, g2, g1_hat, g2_hat):  # sort g1_hat, g2_hat based on g1, g2
    K = np.max(g1) + 1
    H = np.max(g2) + 1
    K0 = np.max(g1_hat) + 1
    H0 = np.max(g2_hat) + 1

    R1 = np.zeros((K, K0))
    for c1 in range(K):
        for c2 in range(K0):
            R1[c1, c2] = len(set(np.where(g1 == c1)[0]) & set(np.where(g1_hat == c2)[0]))

    order1 = np.argsort(np.argmax(R1, axis=0))

    R2 = np.zeros((H, H0))
    for c1 in range(H):
        for c2 in range(H0):
            R2[c1, c2] = len(set(np.where(g2 == c1)[0]) & set(np.where(g2_hat == c2)[0]))

    order2 = np.argsort(np.argmax(R2, axis=0))

    # sort g1_hat, g2_hat
    g1_hat_sort = np.zeros((g1_hat.shape[0]), dtype=np.int64)
    g2_hat_sort = np.zeros((g2_hat.shape[0]), dtype=np.int64)
    for c1 in range(K0):
        nodes_c1 = np.where(g1_hat == order1[c1])
        g1_hat_sort[nodes_c1[0]] = c1

    for c2 in range(H0):
        nodes_c2 = np.where(g2_hat == order2[c2])
        g2_hat_sort[nodes_c2[0]] = c2

    return g1_hat_sort, g2_hat_sort

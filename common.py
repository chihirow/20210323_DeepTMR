import numpy as np
import math
import torch


def define_xy(A, idx_entry, n_batch, device):
    n = A.shape[0]
    p = A.shape[1]
    idx_r = idx_entry // p
    idx_c = idx_entry % p
    x_row_cpu = np.zeros((n_batch, p))
    x_col_cpu = np.zeros((n_batch, n))
    y = np.zeros((n_batch, 1))
    for i in range(n_batch):
        x_row_cpu[i, :] = np.copy(A[idx_r[i], :])  # Row information except for (i, j)
        # x_row_cpu[i, idx_c[i]] = 0  # Delete information of (i, j)th entry
        x_col_cpu[i, :] = np.copy(A[:, idx_c[i]])  # Column information except for (i, j)
        # x_col_cpu[i, idx_r[i]] = 0
        y[i, :] = np.copy(A[idx_r[i], idx_c[i]])
    x_row = torch.tensor(x_row_cpu).to(device, dtype=torch.float)  # n_batch x n
    x_col = torch.tensor(x_col_cpu).to(device, dtype=torch.float)  # n_batch x p
    y = torch.tensor(y).to(device, dtype=torch.float)  # n_batch

    return x_row, x_col, y


def calc_result(A, model, device):
    n = A.shape[0]
    p = A.shape[1]
    A_out = np.zeros((n, p))
    h_row = np.zeros(n)  # Latent feature of rows
    h_col = np.zeros(p)
    for i in range(n):
        for j in range(p):
            x_row_cpu = np.copy(A[i, :])
            # x_row_cpu[j] = 0  # 2021/6/10 delete
            x_row_cpu = x_row_cpu[np.newaxis, :]
            x_col_cpu = np.copy(A[:, j])
            # x_col_cpu[i] = 0  # 2021/6/10 delete
            x_col_cpu = x_col_cpu[np.newaxis, :]
            x_row = torch.tensor(x_row_cpu).to(device, dtype=torch.float)
            x_col = torch.tensor(x_col_cpu).to(device, dtype=torch.float)
            with torch.no_grad():
                A_out_ij, h_row_i, h_col_j = model(x_row, x_col)
                A_out[i, j] = A_out_ij.detach().cpu().clone().numpy()
                if j == 0:
                    h_row[i] = h_row_i.detach().cpu().clone().numpy()
                if i == 0:
                    h_col[j] = h_col_j.detach().cpu().clone().numpy()
    order_row = np.argsort(h_row)
    order_col = np.argsort(h_col)

    return A_out, h_row, h_col, order_row, order_col


def mr_pca_row(A):  # [Friendly2002]
    n = A.shape[0]
    p = A.shape[1]
    A_copy = np.copy(A)
    for i in range(n):
        A_copy[i, :] = (A_copy[i, :] - np.mean(A_copy[i, :])) / np.std(A_copy[i, :] + 1e-5)
    R = (1 / p) * np.matmul(A_copy, A_copy.T)
    _, _, U = np.linalg.svd(R, full_matrices=True)
    u1 = U[:, 0]
    u2 = U[:, 1]
    alpha = np.zeros(n)  # -pi/2 <= <= 3/2 pi
    for i in range(n):
        if u1[i] <= 0:
            alpha[i] = np.arctan(u2[i] / (u1[i] + 1e-5)) + math.pi
        else:
            alpha[i] = np.arctan(u2[i] / (u1[i] + 1e-5))
    idx = np.argsort(alpha)
    alpha_sort = alpha[idx]
    d_alpha = np.zeros(n)
    for i in range(n):
        if i == 0:
            d_alpha[i] = 2 * math.pi + alpha_sort[i] - alpha_sort[i - 1]
        else:
            d_alpha[i] = alpha_sort[i] - alpha_sort[i - 1]
    idx_max = np.argmax(d_alpha)
    idx[idx < idx_max] += n
    idx -= idx_max

    return idx  # row order, A[idx, :]


def mr_svd(A):
    U, D, V = np.linalg.svd(A, full_matrices=True)
    u1 = U[0, :]
    v1 = V[:, 0]
    idx_row = np.argsort(u1)
    idx_col = np.argsort(v1)
    A_approx = D[0] * np.matmul(u1[:, np.newaxis], v1[np.newaxis, :])  # rank-one approximation of A

    return idx_row, idx_col, A_approx


def mr_mds_row(A):
    n = A.shape[0]
    D2 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D2[i, j] = np.sum((A[i, :] - A[j, :]) ** 2)
    J = np.eye(n) - (1 / n) * np.ones((n, n))
    B = - (1 / 2) * np.matmul(J, np.matmul(D2, J))
    w, V = np.linalg.eig(B)
    idx = np.argsort(-w.real)
    v1 = V[:, idx[0]].real
    idx_row = np.argsort(v1)

    return idx_row


def select_order(order_row, order_col, P, P_bar):
    n = P.shape[0]
    p = P.shape[1]
    order_row0 = np.copy(order_row)
    order_row_list = np.append(order_row0[np.newaxis, :], np.flip(order_row0)[np.newaxis, :], axis=0)
    order_col0 = np.copy(order_col)
    order_col_list = np.append(order_col0[np.newaxis, :], np.flip(order_col0)[np.newaxis, :], axis=0)
    err0 = np.inf
    for i in range(2):
        for j in range(2):
            err = np.sum((P_bar - P[order_row_list[i], :][:, order_col_list[j]]) ** 2)
            if err < err0:
                i_order = i
                j_order = j
                err0 = err
    if i_order == 0:
        order_row_opt = order_row0
    else:
        order_row_opt = np.flip(order_row0)
    if j_order == 0:
        order_col_opt = order_col0
    else:
        order_col_opt = np.flip(order_col0)

    err = np.sum((P_bar - P[order_row_opt, :][:, order_col_opt]) ** 2) / (n * p)

    return order_row_opt, order_col_opt, err

"""
@author: Yoni Choukroun, choukroun.yoni@gmail.com
Error Correction Code Transformer
https://arxiv.org/abs/2203.14966
"""

import numpy as np
import torch
import os


def rayleigh_channel(bits, snr_db=5, scale=1 / np.sqrt(2), seed=None):
    if seed is not None:
        np.random.seed(seed)

    bits = np.asarray(bits).astype(int)

    s = 1 - 2 * bits

    r = np.random.rayleigh(scale=scale, size=bits.shape)

    snr_lin = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_lin))

    n = np.random.normal(0, noise_std, size=bits.shape)
    y = r * s + n
    s_hat = np.sign(y)
    s_hat[s_hat == 0] = 1

    bits_hat = (s_hat < 0).astype(int)

    return np.abs(bits_hat - bits)


def Read_pc_matrixrix_alist(fileName):
    with open(fileName, "r") as file:
        lines = file.readlines()
        columnNum, rowNum = np.fromstring(lines[0].rstrip("\n"), dtype=int, sep=" ")
        H = np.zeros((rowNum, columnNum)).astype(int)
        for column in range(4, 4 + columnNum):
            nonZeroEntries = np.fromstring(
                lines[column].rstrip("\n"), dtype=int, sep=" "
            )
            for row in nonZeroEntries:
                if row > 0:
                    H[row - 1, column - 4] = 1
        return H


#############################################
def row_reduce(mat, ncols=None):
    assert mat.ndim == 2
    ncols = mat.shape[1] if ncols is None else ncols
    mat_row_reduced = mat.copy()
    p = 0
    for j in range(ncols):
        idxs = p + np.nonzero(mat_row_reduced[p:, j])[0]
        if idxs.size == 0:
            continue
        mat_row_reduced[[p, idxs[0]], :] = mat_row_reduced[[idxs[0], p], :]
        idxs = np.nonzero(mat_row_reduced[:, j])[0].tolist()
        idxs.remove(p)
        mat_row_reduced[idxs, :] = mat_row_reduced[idxs, :] ^ mat_row_reduced[p, :]
        p += 1
        if p == mat_row_reduced.shape[0]:
            break
    return mat_row_reduced, p


def get_generator(pc_matrix_):
    assert pc_matrix_.ndim == 2
    pc_matrix = pc_matrix_.copy().astype(bool).transpose()
    pc_matrix_I = np.concatenate(
        (pc_matrix, np.eye(pc_matrix.shape[0], dtype=bool)), axis=-1
    )
    pc_matrix_I, p = row_reduce(pc_matrix_I, ncols=pc_matrix.shape[1])
    return row_reduce(pc_matrix_I[p:, pc_matrix.shape[1] :])[0]


def get_standard_form(pc_matrix_):
    pc_matrix = pc_matrix_.copy().astype(bool)
    next_col = min(pc_matrix.shape)
    for ii in range(min(pc_matrix.shape)):
        while True:
            rows_ones = ii + np.where(pc_matrix[ii:, ii])[0]
            if len(rows_ones) == 0:
                new_shift = np.arange(ii, min(pc_matrix.shape) - 1).tolist() + [
                    min(pc_matrix.shape) - 1,
                    next_col,
                ]
                old_shift = np.arange(ii + 1, min(pc_matrix.shape)).tolist() + [
                    next_col,
                    ii,
                ]
                pc_matrix[:, new_shift] = pc_matrix[:, old_shift]
                next_col += 1
            else:
                break
        pc_matrix[[ii, rows_ones[0]], :] = pc_matrix[[rows_ones[0], ii], :]
        other_rows = pc_matrix[:, ii].copy()
        other_rows[ii] = False
        pc_matrix[other_rows] = pc_matrix[other_rows] ^ pc_matrix[ii]
    return pc_matrix.astype(int)


#############################################


def sign_to_bin(x):
    return 0.5 * (1 - x)


def bin_to_sign(x):
    return 1 - 2 * x


def EbN0_to_std(EbN0, rate):
    snr = EbN0 + 10.0 * np.log10(2 * rate)
    return np.sqrt(1.0 / (10.0 ** (snr / 10.0)))


def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()


def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()


def rank(a):
    return np.linalg.matrix_rank(a)


def full_rank(a):
    if rank(a) == min(a.shape):
        return True
    else:
        return False


def find_error(binary_noise):
    column_indices = np.where(np.any(binary_noise == 1, axis=0))[0]
    return column_indices


def BLER_(x_pred, x_gt, generate_matrix):
    error_index = find_error(x_gt.T)
    out = x_pred - x_gt[error_index, :]
    detect = np.sum(out, axis=1)
    error_index_ = detect > 0

    index = np.ones(x_gt.shape[0], dtype=bool)
    index[error_index[error_index_]] = False
    decode_matrix = generate_matrix[index, :]
    return full_rank(decode_matrix)


#############################################

import numpy as np

def _computeW(N, k):
    W = np.zeros(N)

    for i in range(N):
        W[i] = np.cos(k * np.pi * (2 * i + 1) / (2 * N))

    return W

def DCT1(v):
    N = len(v)
    a_coeff = np.zeros(N)

    for k in range(N):
        W_k = _computeW(N, k)
        if k == 0:
            a_coeff[k] = (v @ W_k) / np.sqrt(N)
        else:
            a_coeff[k] = (v @ W_k) / np.sqrt(N / 2)

    return a_coeff

def DCT2(A):

    row, columns = A.shape
    alpha_coeff = np.zeros((row, columns))

    for k in range(row):
        alpha_coeff[k, :] = DCT1(A[k, :])

    for k in range(columns):
        alpha_coeff[:, k] = DCT1(alpha_coeff[:, k])

    return alpha_coeff

def cut_frequences(coeffs, d):

    row, columns = coeffs.shape
    cutted_coeffs = np.zeros((row, columns))

    for k in range(row):
        for l in range(columns):
            if k + l < d:
                cutted_coeffs[k, l] = coeffs[k, l]

    return cutted_coeffs
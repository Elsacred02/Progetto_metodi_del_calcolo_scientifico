import numpy as np
def IDCT1(c):
    N = len(c)
    f = np.zeros(N)

    for j in range(N):

        # Qui esce cos(0) quindi semplifico la scrittura dell'operazione
        s = c[0] * np.sqrt(1 / N)

        for k in range(1, N):
            s += c[k] * np.sqrt(2 / N) * np.cos(np.pi * k * (2 * j + 1) / (2 * N))
        f[j] = s

    return f

def IDCT2(A):
    row, columns = A.shape
    original_matrix = np.zeros((row, columns))

    # inversa sulle colonne
    for k in range(columns):
        original_matrix[:, k] = IDCT1(A[:, k])

    # inversa sulle righe
    for k in range(row):
        original_matrix[k, :] = IDCT1(original_matrix[k, :])

    return original_matrix
import numpy as np
def IDCT1(coeffs):
    N = len(coeffs)
    x = np.zeros(N)

    for n in range(N):

        # Qui esce cos(0) quindi semplifico la scrittura dell'operazione
        s = coeffs[0] * np.sqrt(1/N)

        for k in range(1, N):
            s += coeffs[k] * np.sqrt(2/N) * np.cos(np.pi * k * (2*n+1) / (2*N))
        x[n] = s

    return x

def IDCT2(coeffs):
    row, columns = coeffs.shape
    original_matrix = np.zeros((row, columns))

    # inversa sulle colonne
    for k in range(columns):
        original_matrix[:, k] = IDCT1(coeffs[:, k])

    # inversa sulle righe
    for k in range(row):
        original_matrix[k, :] = IDCT1(original_matrix[k, :])

    return original_matrix
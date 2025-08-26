from secondo_progetto.compression import *
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.fftpack import dct

def main():
    Ns = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
          55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
          105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
    times_custom = []
    times_scipy = []

    for N in Ns:
        A = np.random.rand(N, N)

        # Tempo DCT2 custom
        start = time()
        my_dct = DCT2(A)
        times_custom.append(time() - start)

        # Tempo DCT2 scipy
        start = time()
        library_dct =dct(dct(A.T, norm='ortho').T, norm='ortho')
        times_scipy.append(time() - start)

        # Confronto
        if not np.allclose(my_dct, library_dct):
            print(f"Errore per N={N}")

    # --- Grafico ---
    plt.figure(figsize=(8, 6))
    plt.plot(Ns, times_custom, 'o-', label='DCT2 custom')
    plt.plot(Ns, times_scipy, 's-', label='DCT2 scipy')
    plt.yscale('log')
    plt.xlabel('Dimensione N')
    plt.ylabel('Tempo [s] (scala logaritmica)')
    plt.title('Confronto tempi DCT2')
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)

    # Salvataggio immagine
    plt.savefig('output_immagini/confronto_dct2.png', dpi=300)

    # block = np.array([
    #     [231, 32, 233, 161, 24, 71, 140, 245],
    #     [247, 40, 248, 245, 124, 204, 36, 107],
    #     [234, 202, 245, 167, 9, 217, 239, 173],
    #     [193, 190, 100, 167, 43, 180, 8, 70],
    #     [11, 24, 210, 177, 81, 243, 8, 112],
    #     [97, 195, 203, 47, 125, 114, 165, 181],
    #     [193, 70, 174, 167, 41, 30, 127, 245],
    #     [87, 149, 57, 192, 65, 129, 178, 228]
    # ], dtype=float)
    #
    # print(DCT2(block))
    # print(IDCT2(DCT2(block)))

if __name__ == "__main__":
    main()
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
        DCT2(A)
        times_custom.append(time() - start)

        # Tempo DCT2 scipy
        start = time()
        dct(dct(A.T, norm='ortho').T, norm='ortho')
        times_scipy.append(time() - start)

    # --- Grafico ---
    plt.figure(figsize=(8, 6))
    plt.plot(Ns, times_custom, 'o-', label='DCT2 fatta in casa')
    plt.plot(Ns, times_scipy, 's-', label='DCT2 scipy')
    plt.yscale('log')
    plt.xlabel('Dimensione N')
    plt.ylabel('Tempo [s] (scala logaritmica)')
    plt.title('Confronto tempi DCT2')
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)

    # Salvataggio immagine
    plt.savefig('img/confronto_dct2.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
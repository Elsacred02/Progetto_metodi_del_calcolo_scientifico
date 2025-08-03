from solutori_lineari_iterativi.solvers import *
from scipy.io import mmread
import numpy as np
import matplotlib.pyplot as plt
import os

def main():

    file_name = "vem2"
    path = f"primo_progetto/matrici_test/{file_name}.mtx"

    # Carica matrice e dati iniziali
    matrix = mmread(path)
    A = matrix.toarray()
    x_true = np.ones(A.shape[0])
    b = A @ x_true
    tol_values = [1e-4, 1e-6, 1e-8, 1e-10]

    methods = {
        'conjugated-gradient': coniugate_gradient_solver,
        'gauss-seidel': gauss_seidel_solver,
        'gradient': gradient_solver,
        'jacobi': jacobi_solver
    }

    # Dizionari: tol → [valori per ogni metodo]
    iterations = {tol: [] for tol in tol_values}
    errors = {tol: [] for tol in tol_values}
    times = {tol: [] for tol in tol_values}

    for method_name, solver in methods.items():
        for tol in tol_values:
            x0 = np.ones_like(x_true)
            sol, k, rel_err, elapsed = solver(A, b, x0, tol)
            iterations[tol].append(k)
            errors[tol].append(rel_err)
            times[tol].append(elapsed)

    method_names = list(methods.keys())
    x = np.arange(len(method_names))
    width = 0.2
    colors = ['#4B0082', '#008080', '#FFD700', '#CCCC00']
    tol_labels = [f"tol={t:.0e}" for t in tol_values]

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    file_subtitle = f"Statistiche per la matrice: {file_name}"
    fig.suptitle(file_subtitle, fontsize=16)

    # Iterazioni
    for i, tol in enumerate(tol_values):
        axs[0].bar(x + i * width, iterations[tol], width, label=tol_labels[i], color=colors[i])
    axs[0].set_title("Iterazioni")
    axs[0].set_xticks(x + width * 1.5)
    axs[0].set_xticklabels(method_names, rotation=15)
    axs[0].set_yscale('log')
    axs[0].set_ylabel("Iterazioni")

    # Errore
    for i, tol in enumerate(tol_values):
        axs[1].bar(x + i * width, errors[tol], width, label=tol_labels[i], color=colors[i])
    axs[1].set_title("Errore relativo")
    axs[1].set_xticks(x + width * 1.5)
    axs[1].set_xticklabels(method_names, rotation=15)
    axs[1].set_yscale('log')
    axs[1].set_ylabel("Errore")

    # Tempo
    for i, tol in enumerate(tol_values):
        axs[2].bar(x + i * width, times[tol], width, label=tol_labels[i], color=colors[i])
    axs[2].set_title("Tempo (s)")
    axs[2].set_xticks(x + width * 1.5)
    axs[2].set_xticklabels(method_names, rotation=15)
    axs[2].set_yscale('log')
    axs[2].set_ylabel("Tempo")

    # Legenda unica
    fig.legend(loc='lower center', ncol=len(tol_values), bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Salvataggio
    os.makedirs("primo_progetto/matrici_test/output_immagini", exist_ok=True)

    image_path = f"primo_progetto/matrici_test/output_immagini/statistiche_{file_name}.png"
    plt.savefig(image_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    main()

from primo_progetto.solutori_lineari_iterativi import *
from scipy.io import mmread
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def main():

    file_name = "spa1"
    path = f"matrici_test/{file_name}.mtx"

    # Carica matrice e dati iniziali
    matrix = mmread(path)
    A = matrix.toarray()
    x = np.ones(A.shape[0])
    b = A @ x
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
            sol, k, rel_err, elapsed = solver(A, b, x, tol)
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

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(tol_values), bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])

    # Salvataggio
    os.makedirs("output_immagini", exist_ok=True)
    image_path = f"output_immagini/statistiche_{file_name}.png"
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Tabella con i dati
    table_rows = []

    method_nomi = {
        'jacobi': 'Jacobi',
        'gauss-seidel': 'Gauss–Seidel',
        'gradient': 'Gradiente',
        'conjugated-gradient': 'Gradiente Coniugato'
    }

    for tol in tol_values:
        first_row = True
        for method_key in methods.keys():
            row = {
                "Tolleranza": f"{tol:.0e}" if first_row else "",
                "Metodo": method_nomi[method_key],
                "Iterazioni": f"{iterations[tol][list(methods).index(method_key)]}",
                "Tempo (s)": f"{times[tol][list(methods).index(method_key)]:.5f}",
                "Errore relativo": f"{errors[tol][list(methods).index(method_key)]:.2e}"
            }
            table_rows.append(row)
            first_row = False

    df = pd.DataFrame(table_rows)

    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.35 * len(df)))
    ax.axis('off')

    plt.rcParams["font.family"] = "serif"

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.2)

    plt.title(f"Table 1: Risultati per $\\mathtt{{{file_name}}}$.", fontsize=14, pad=20)

    os.makedirs("output_immagini", exist_ok=True)
    plt.savefig(f"output_immagini/tabella_{file_name}.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()

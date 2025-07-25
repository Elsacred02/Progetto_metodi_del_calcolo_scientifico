from solutori_lineari_iterativi.solvers import *
import numpy as np

def main():
    # Matrice A (esempio semplice: diagonale dominante)
    A = np.array([[4.0, 1.0],
                  [2.0, 3.0]])

    # Vettore termine noto b
    b = np.array([1.0, 2.0])

    # Vettore iniziale x0 (stima iniziale)
    x = np.linalg.solve(A, b)

    # Tolleranza per il criterio di arresto
    tol = 1e-6

    sol, k, relative_error, elapsed_time = jacobi_solver(A, b, x, tol)

    print(sol, k, relative_error, elapsed_time)

if __name__ == '__main__':
    main()
import numpy as np
import time

from debugpy.common.timestamp import current


def _base_iterative_solver(A, b, x, tol, update_method):

    # Controlli preliminari sull'input
    rows, cols = A.shape
    if rows != cols:
        print("[Errore] : la matrice A inserita non è quadrata")
        return None, 0, 0, 0
    if rows != len(b):
        print("[Errore] : il vettore b inserito non ha la stessa dimensione delle righe della matrice A")
        return None, 0, 0, 0
    if rows != len(x):
        print("[Errore] : il vettore x inserito non ha la stessa dimensione delle righe della matrice A")
        return None, 0, 0, 0

    #Avvio cronometro
    start_time = time.time()

    # Inizializzo la soluzione iniziale ed il contatore delle iterazioni
    sol = np.zeros(rows)
    k = 0

    while (np.linalg.norm(A @ sol - b, ord=np.inf) / np.linalg.norm(b)) > tol:
        sol = update_method(sol)
        k = k + 1
        if k > 20000:
            print("[Errore] : il metodo non converge")
            return None, 0, 0, 0

    # Fermo cronometro
    elapsed_time = time.time() - start_time

    relative_error = np.linalg.norm(A @ sol - b, ord=np.inf) / np.linalg.norm(b, ord=np.inf)

    return sol, k, relative_error, elapsed_time


def jacobi_solver(A, b, x, tol):

    P = np.diag(np.diag(A))
    P_inv = np.linalg.inv(P)
    def _jacobi_update(current_sol):
        new_sol = current_sol + P_inv @ (b - A @ current_sol)
        return new_sol

    return _base_iterative_solver(A, b, x, tol, _jacobi_update)



def gauss_seidel_solver(A, b, x, tol):

    P = np.tril(A)
    def _gauss_seidel_update(current_sol):
        r = b - A @ current_sol
        y = _linear_solve_forward(P, r)
        return current_sol + y

    return _base_iterative_solver(A, b, x, tol, _gauss_seidel_update)



def _linear_solve_forward(L, b):

    rows, cols = L.shape
    if rows != cols:
        print("[Errore] : la matrice A inserita non è quadrata")
        return None
    if rows != len(b):
        print("[Errore] : il vettore b inserito non ha la stessa dimensione delle righe della matrice A")
        return None

    if L[0,0] == 0:
        return "[Errore] : impossibile risolvere il sistema"

    x = np.zeros(rows, dtype=float)
    x[0] = b[0] / L[0, 0]
    for i in range(1, rows):
        if L[i, i] == 0:
            return "[Errore] : impossibile risolvere il sistema"
        else:
            x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]

    return x

def gradient_solver(A, b, x, tol):
    def _gradient_update(current_sol):
        r = b - A @ current_sol
        y = A @ r
        numerator = r.T @ r
        denominator = r.T @ y
        alpha = numerator / denominator
        return current_sol + alpha * r

    return _base_iterative_solver(A, b, x, tol, _gradient_update)



def coniugate_gradient_solver(A, b, x, tol):

    r = b - A @ np.zeros(A.shape[0])
    current_d = r

    def _coniugate_gradient_update(current_sol):
        nonlocal current_d
        r = b - A @ current_sol
        y = A @ current_d
        alpha = (current_d @ r) / (current_d @ y)
        new_sol = current_sol + alpha * current_d

        new_r = b - A @ new_sol
        w = A @ new_r
        beta = (current_d @ w) / (current_d @ y)
        current_d = new_r - beta * current_d

        return new_sol

    return _base_iterative_solver(A, b, x, tol, _coniugate_gradient_update)




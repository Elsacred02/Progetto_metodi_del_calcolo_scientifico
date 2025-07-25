import numpy as np
import time

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



def gauss_seidel_update(A, current_sol, b):
    print("Ciao sono gauss seidel")
    print(A)
    print(current_sol)
    print(b)

def gradient_update(A, current_sol, b):
    print("Ciao sono gradient")
    print(A)
    print(current_sol)

def coniugate_gradient_update(A, current_sol, b):
    print("Ciao sono coniugate gradient")
    print(A)
    print(current_sol)
    print(b)
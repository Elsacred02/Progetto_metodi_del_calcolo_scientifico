import numpy as np
import time

from PyQt5.QtQml import kwargs


def base_iterative_solver(A, b, x, tol, update_method):

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

    # Inizializzo la soluzione iniziale ed il contatore delle iterazioni
    sol = np.zeros(A.shape[0])
    k = 0

    while (np.linalg.norm(np.dot(A, sol) - b) / np.linalg.norm(b)) > tol:
        sol = update_method(A, b, sol)



def _jacobi_update(A, b, current_sol):
    print("Ciao sono jacobi")

def _gauss_seidel_update(A, b, current_sol):
    print("Ciao sono gauss seidel")

def _gradient_update(A, current_sol, **kwargs):
    print("Ciao sono gradient")

def _coniugate_gradient_update(A, current_sol, b):
    print("Ciao sono coniugate gradient")
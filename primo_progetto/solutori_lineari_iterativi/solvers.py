import numpy as np
import time

from debugpy.common.timestamp import current


def _base_iterative_solver(A, b, x, tol, update_method):

    """
    Funzione base per la risoluzione di un sistema lineare Ax=b utilizzando metodi iterativi.

    Parameters
    ----------
    A: numpy array
        Matrice dei coefficienti del sistema lineare
    b: numpy array
        Vettore dei termini noti del sistema lineare
    x: numpy array
        Vettore soluzione esatta del sistema lineare (utilizzata per calcolare l'errore relativo)
    tol: float
        Tolleranza per la convergenza del metodo
    update_method: funzione
        Funzione che implementa il metodo iterativo scelto per la risoluzione del sistema

    Returns
    -------
    sol: numpy array
        Vettore soluzione approssimata del sistema lineare
    k: int
        Numero di iterazioni eseguite
    relative_error: float
        Errore relativo tra la soluzione esatta e quella approssimata
    elapsed_time: float
        Tempo impiegato per risolvere il sistema
    """

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

    # Struttura base dei metodi iterativi
    while (np.linalg.norm(A @ sol - b, ord=np.inf) / np.linalg.norm(b)) > tol:
        sol = update_method(sol)
        k = k + 1
        if k > 20000:
            print("[Errore] : il metodo non converge")
            return None, 0, 0, 0

    # Fermo cronometro
    elapsed_time = time.time() - start_time

    # Calcolo l'errore relativo
    relative_error = np.linalg.norm(sol - x, ord=np.inf) / np.linalg.norm(x, ord=np.inf)

    # Restituisco i parametri calcolati
    return sol, k, relative_error, elapsed_time


def jacobi_solver(A, b, x, tol):
    """
    Risolve il sistema lineare A @ x = b utilizzando il metodo di Jacobi.

    Parameters
    ----------
    A: numpy array
        Matrice dei coefficienti del sistema lineare
    b: numpy array
        Vettore dei termini noti del sistema lineare
    x: numpy array
        Vettore soluzione esatta del sistema lineare (utilizzata per calcolare l'errore relativo)
    tol: float
        Tolleranza per la convergenza del metodo

    Returns
    -------
    sol: numpy array
        Vettore soluzione approssimata del sistema lineare
    k: int
        Numero di iterazioni eseguite
    relative_error: float
        Errore relativo tra la soluzione esatta e quella approssimata
    elapsed_time: float
        Tempo impiegato per risolvere il sistema
    """

    if not _is_symmetric_positive_definite(A):
        print("[Warning] - la matrice A non è simmetrica e definita positiva")
    if not _is_row_diagonally_dominant(A):
        print("[Warning] - la matrice A non ha dominanza triangolare per righe, la convergenza non e' garantita")

    # Definisco la matrice P secondo il metodo di Jacobi e calcolo l'inversa
    P = np.diag(np.diag(A))

    if np.any(np.diag(A) == 0):
        print("[Errore] - sulla diagonale di A è presente un elemento nullo, quindi la matrice P non e' invertibile.")
        return None, 0, 0, 0

    P_inv = np.linalg.inv(P)

    # Effettua l'aggiornamento della soluzione corrente usando il metodo di Jacobi
    def _jacobi_update(current_sol):
        new_sol = current_sol + P_inv @ (b - A @ current_sol)
        return new_sol

    return _base_iterative_solver(A, b, x, tol, _jacobi_update)



def gauss_seidel_solver(A, b, x, tol):

    """
    Risolve il sistema lineare A @ x = b utilizzando il metodo di Gauss-Seidel.

    Parameters
    ----------
    A: numpy array
        Matrice dei coefficienti del sistema lineare
    b: numpy array
        Vettore dei termini noti del sistema lineare
    x: numpy array
        Vettore soluzione esatta del sistema lineare (utilizzata per calcolare l'errore relativo)
    tol: float
        Tolleranza per la convergenza del metodo

    Returns
    -------
    sol: numpy array
        Vettore soluzione approssimata del sistema lineare
    k: int
        Numero di iterazioni eseguite
    relative_error: float
        Errore relativo tra la soluzione esatta e quella approssimata
    elapsed_time: float
        Tempo impiegato per risolvere il sistema
    """
    if not _is_symmetric_positive_definite(A):
        print("[Warning] - la matrice A non è simmetrica e definita positiva")
    if not _is_row_diagonally_dominant(A):
        print("[Warning] - la matrice A non ha dominanza triangolare per righe, la convergenza non e' garantita")

    # Definisco la matrice P secondo il metodo di Gauss-Seidel
    P = np.tril(A)

    # Effettua l'aggiornamento della soluzione corrente usando il metodo di Gauss-Seidel
    def _gauss_seidel_update(current_sol):
        r = b - A @ current_sol
        y = _linear_solve_forward(P, r)
        return current_sol + y

    return _base_iterative_solver(A, b, x, tol, _gauss_seidel_update)



def _linear_solve_forward(L, b):

    """
    Risolve il sistema lineare L @ x = b, dove L è una matrice inferiore triangolare,
    utilizzando l'algoritmo di sostituzione in avanti.

    Parameters
    ----------
    L: numpy array
        Matrice dei coefficienti del sistema lineare
    b: numpy array
        Vettore dei termini noti del sistema lineare

    Returns
    -------
    sol: numpy array
        Vettore soluzione approssimata del sistema lineare
    """

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
    """
    Risolve il sistema lineare A @ x = b utilizzando il metodo del gradiente.

    Parameters
    ----------
    A: numpy array
        Matrice dei coefficienti del sistema lineare
    b: numpy array
        Vettore dei termini noti del sistema lineare
    x: numpy array
        Vettore soluzione esatta del sistema lineare (utilizzata per calcolare l'errore relativo)
    tol: float
        Tolleranza per la convergenza del metodo

    Returns
    -------
    sol: numpy array
        Vettore soluzione approssimata del sistema lineare
    k: int
        Numero di iterazioni eseguite
    relative_error: float
        Errore relativo tra la soluzione esatta e quella approssimata
    elapsed_time: float
        Tempo impiegato per risolvere il sistema
    """

    if not _is_symmetric_positive_definite(A):
        print("[Errore] - la matrice A non è simmetrica e definita positiva")
        return None, 0, 0, 0
    # Effettua l'aggiornamento della soluzione corrente usando il metodo del gradiente
    def _gradient_update(current_sol):
        r = b - A @ current_sol
        y = A @ r
        numerator = r.T @ r
        denominator = r.T @ y
        alpha = numerator / denominator
        return current_sol + alpha * r

    return _base_iterative_solver(A, b, x, tol, _gradient_update)

def coniugate_gradient_solver(A, b, x, tol):
    """
    Risolve il sistema lineare A @ x = b utilizzando il metodo del gradiente coniugato.

    Parameters
    ----------
    A: numpy array
        Matrice dei coefficienti del sistema lineare
    b: numpy array
        Vettore dei termini noti del sistema lineare
    x: numpy array
        Vettore soluzione esatta del sistema lineare (utilizzata per calcolare l'errore relativo)
    tol: float
        Tolleranza per la convergenza del metodo

    Returns
    -------
    sol: numpy array
        Vettore soluzione approssimata del sistema lineare
    k: int
        Numero di iterazioni eseguite
    relative_error: float
        Errore relativo tra la soluzione esatta e quella approssimata
    elapsed_time: float
        Tempo impiegato per risolvere il sistema
    """

    if not _is_symmetric_positive_definite(A):
        print("[Errore] - la matrice A non è simmetrica e definita positiva")
        return None, 0, 0, 0

    # Inizializzazione delle variabili
    r = b - A @ np.zeros(A.shape[0])
    current_d = r

    # Effettua l'aggiornamento della soluzione corrente usando il metodo del gradiente coniugato
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

def _is_symmetric(A, tol=1e-8):
    return np.allclose(A, A.T, atol=tol)

def _is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def _is_symmetric_positive_definite(A, tol=1e-8):
    return _is_symmetric(A, tol) and _is_positive_definite(A)

def _is_row_diagonally_dominant(A):
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_diag_sum = np.sum(np.abs(A[i, :])) - diag
        if diag <= off_diag_sum:
            return False
    return True

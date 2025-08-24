import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
from scipy.fftpack import dct, idct
from compression import *
import os

def main():
    percorso_file = {"path": None}  # uso un dizionario per renderlo modificabile dentro le funzioni

    def scegli_file():
        percorso = filedialog.askopenfilename(
            title="Seleziona un file BMP",
            filetypes=[("Immagini BMP", "*.bmp")]  # filtro solo BMP
        )
        if percorso:
            percorso_file["path"] = percorso
            etichetta_file.config(text=f"Hai scelto: {percorso}")
        else:
            etichetta_file.config(text="Nessun file selezionato")
            percorso_file["path"] = None

    def comprimi_immagine():
        # --- 1. Controllo se è stato scelto un file ---
        if not percorso_file["path"]:
            messagebox.showerror("Errore", "Devi selezionare un file BMP prima di procedere.")
            return

        # --- 2. Recupero e controllo input numerici ---
        try:
            valore_F = int(entry_F.get())
            valore_d = int(entry_d.get())
        except ValueError:
            messagebox.showerror("Errore", "I valori di F e d devono essere numeri interi.")
            return

        # --- 3. Carico immagine per dimensioni ---
        try:
            img = Image.open(percorso_file["path"])
            larghezza, altezza = img.size
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile aprire l'immagine: {e}")
            return

        # --- 4. Controllo vincoli su F ---
        if valore_F >= larghezza or valore_F >= altezza:
            messagebox.showerror("Errore", f"F deve essere minore delle dimensioni dell'immagine "
                                           f"(larghezza={larghezza}, altezza={altezza}).")
            return

        # --- 5. Controllo vincoli su d ---
        if not (0 <= valore_d <= 2 * valore_F - 2):
            messagebox.showerror("Errore", f"d deve essere compreso tra 0 e {2 * valore_F - 2}.")
            return

        # --- 6. Conversione immagine in matrice ---
        img_gray = img.convert("L")
        image_as_matrix = np.array(img_gray)

        # --- 7. Effettuo la compressione ---
        row, columns = image_as_matrix.shape

        for i in range(0, row, valore_F):

            for j in range(0, columns, valore_F):

                if valore_F + 8 < row and valore_F + 8 < columns:

                    sub_matrix = image_as_matrix[i:i+valore_F, j:j+valore_F]
                    sub_matrix_dct = dct(dct(sub_matrix.T, norm='ortho').T, norm='ortho')
                    cutted_dct_submatrix = cut_frequences(sub_matrix_dct, valore_d)
                    idct_submatrix = idct(idct(cutted_dct_submatrix.T, norm='ortho').T, norm='ortho')
                    idct_submatrix = np.round(idct_submatrix)
                    idct_submatrix = np.clip(idct_submatrix, 0, 255)
                    image_as_matrix[i:i+valore_F, j:j+valore_F] = idct_submatrix

        matrice_uint8 = np.round(image_as_matrix).astype(np.uint8)
        img_compressa = Image.fromarray(matrice_uint8)

        # --- 8. Visualizzazione immagine originale e compressa ---
        margine = 20  # pixel tra le due immagini
        larghezza_tot = img_gray.width + img_compressa.width + margine
        altezza_max = max(img_gray.height, img_compressa.height)
        img_affiancata = Image.new("L", (larghezza_tot, altezza_max), color=255)  # sfondo bianco
        img_affiancata.paste(img_gray, (0, 0))
        img_affiancata.paste(img_compressa, (img_gray.width + margine, 0))
        img_affiancata.show()

        # --- 9. Salvataggio immagine ---
        output_dir = "compressed_images"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "immagine_compressa.bmp")
        img_compressa.save(output_path)
        print(f"Immagine compressa salvata in: {output_path}")

    # Finestra principale
    finestra = tk.Tk()
    finestra.title("Selezione file BMP e parametri")
    finestra.geometry("450x350")

    # Sezione file
    etichetta_file = tk.Label(finestra, text="Nessun file selezionato", font=("Arial", 12))
    etichetta_file.pack(pady=10)

    bottone_file = tk.Button(finestra, text="Scegli un file BMP", command=scegli_file)
    bottone_file.pack(pady=5)

    # Sezione valori F e d
    frame_parametri = tk.Frame(finestra)
    frame_parametri.pack(pady=20)

    # Valore F
    label_F = tk.Label(frame_parametri, text="Valore F:", font=("Arial", 12))
    label_F.grid(row=0, column=0, padx=5, pady=5, sticky="e")
    entry_F = tk.Entry(frame_parametri, width=10)
    entry_F.grid(row=0, column=1, padx=5, pady=5)

    # Valore d
    label_d = tk.Label(frame_parametri, text="Valore d:", font=("Arial", 12))
    label_d.grid(row=1, column=0, padx=5, pady=5, sticky="e")
    entry_d = tk.Entry(frame_parametri, width=10)
    entry_d.grid(row=1, column=1, padx=5, pady=5)

    # Bottone per effettuare la compressione i valori
    bottone_valori = tk.Button(finestra, text="Comprimi immagine", command=comprimi_immagine)
    bottone_valori.pack(pady=10)

    # Etichetta di output
    etichetta_risultato = tk.Label(finestra, text="", font=("Arial", 12))
    etichetta_risultato.pack(pady=10)

    # Avvio loop
    finestra.mainloop()

if __name__ == "__main__":
    main()

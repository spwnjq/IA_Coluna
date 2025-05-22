import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Carregamento do modelo
model = load_model("coluna.h5")

# Função de previsão
def classificar_imagem():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = image.load_img(file_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)[0][0]
        classe = "Fratura" if prediction > 0.5 else "Normal"
        confiança = prediction if prediction > 0.5 else 1 - prediction
        
        # Exibição da imagem e resultado
        img_display = Image.open(file_path).resize((200, 200))
        img_tk = ImageTk.PhotoImage(img_display)
        panel.configure(image=img_tk)
        panel.image = img_tk
        resultado_label.config(text=f"Classe: {classe} ({confiança:.2%})")

# Interface
root = tk.Tk()
root.title("Classificador de Coluna Cervical")

btn = tk.Button(root, text="Selecionar Imagem", command=classificar_imagem)
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

resultado_label = tk.Label(root, text="", font=("Arial", 14))
resultado_label.pack(pady=10)

root.mainloop()

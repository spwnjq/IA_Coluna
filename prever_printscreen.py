import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import ImageGrab
import time

# Carrega o modelo treinado
model = load_model("coluna.h5")

# Aguarda 3 segundos para o usuário posicionar a tela
print("📸 Prepare sua tela. Captura em 3 segundos...")
time.sleep(3)

# Captura a tela inteira
screenshot = ImageGrab.grab()

# Redimensiona para o tamanho usado no modelo
screenshot = screenshot.resize((150, 150))
img_array = image.img_to_array(screenshot) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Previsão
prediction = model.predict(img_array)[0][0]
classe = "Fratura" if prediction > 0.5 else "Normal"
confiança = prediction if prediction > 0.5 else 1 - prediction

print(f"🧠 Classe prevista: {classe} ({confiança:.2%} de confiança)")

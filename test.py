from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# 1. Carrega o modelo já treinado
model = load_model("coluna.h5")
print("✅ Modelo carregado.")

# 2. (Opcional) Avaliar no conjunto de validação, se disponível
val_dir = "modelo/modelo/val"  # Caminho para os dados de validação, se existirem
img_height, img_width = 224, 224
batch_size = 32

if os.path.exists(val_dir):
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary'
    )

    loss, acc = model.evaluate(val_generator)
    print(f"📊 Acurácia no conjunto de validação: {acc:.2%}")
else:
    print("⚠️ Conjunto de validação não encontrado. Avaliação pulada.")

# 3. Teste com uma imagem individual
imagem_teste = "raiox.png"  # Substitua pelo nome da imagem que deseja testar

if os.path.exists(imagem_teste):
    img = image.load_img(imagem_teste, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    pred = model.predict(img_array)
    classe_predita = 'Anormal' if pred[0][0] > 0.5 else 'Normal'
    confianca = pred[0][0] if pred[0][0] > 0.5 else 1 - pred[0][0]

    print(f"🩻 Classe predita: {classe_predita} (Confiança: {confianca:.2%})")
else:
    print("⚠️ Imagem de teste 'raiox.png' não encontrada.")

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

def preprocess_image(image_array, target_size=(224, 224)):
    """Pré-processa a imagem recebida como array para o modelo."""
    if not isinstance(image_array, np.ndarray):
        raise ValueError("A imagem deve ser um array numpy")

    image = Image.fromarray(image_array.astype('uint8'))
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalização
    image = np.expand_dims(image, axis=0)
    return image

def image_predict(model_path, image_array, class_names=None):
    """
    Faz a predição com um modelo multiclasse ou binário.
    
    Args:
        model_path (str): Caminho para o modelo .h5
        image_array (np.ndarray): Imagem em formato array (RGB)
        class_names (list[str], opcional): Lista com os nomes das classes.

    Returns:
        str | int: Nome da classe prevista ou índice da classe.
    """
    model = load_model(model_path)
    image = preprocess_image(image_array)
    prediction = model.predict(image)

    if prediction.shape[1] == 1:
        # Binário
        return int(prediction[0][0] >= 0.5)
    else:
        # Multiclasse
        class_index = np.argmax(prediction[0])
        if class_names:
            return class_names[class_index]
        return int(class_index)

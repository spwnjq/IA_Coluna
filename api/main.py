# api/main.py
from fastapi import FastAPI, UploadFile, File
from predict_utils import image_predict
from PIL import Image
import numpy as np
import io

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_array = np.array(image)

    # Caminho do modelo salvo
    model_path = "models/coluna.h5"
    prediction = image_predict(model_path, image_array)

    return {"predicao": prediction}

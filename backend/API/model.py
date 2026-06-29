import tensorflow as tf
import numpy as np
import pickle

from PIL import Image
from io import BytesIO
from fastapi import UploadFile

# Load Model
model = tf.keras.models.load_model("models/fruit_classifier.keras")

# Load Class Names
with open("models/class_names.pkl", "rb") as f:
    class_names = pickle.load(f)


async def predict_image(file: UploadFile):

    image = Image.open(BytesIO(await file.read()))

    image = image.convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0][0]

    fresh_probability = (1 - prediction) * 100
    rotten_probability = prediction * 100

    if prediction >= 0.5:
        label = "Rotten"
        confidence = rotten_probability
    else:
        label = "Fresh"
        confidence = fresh_probability

    return {

        "prediction": label,

        "confidence": round(float(confidence), 2),

        "fresh_probability": round(float(fresh_probability), 2),

        "rotten_probability": round(float(rotten_probability), 2)

    }
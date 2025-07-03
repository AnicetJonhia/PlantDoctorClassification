
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import sys

import os

model = load_model("plant_model.h5")

class_names = sorted([d for d in os.listdir("dataset_example") if os.path.isdir(os.path.join("dataset_example", d))])

def predict(img_path):
    img = Image.open(img_path).resize((224,224))
    arr = np.expand_dims(np.array(img)/255.0, axis=0)
    pred = model.predict(arr)[0]
    idx = np.argmax(pred)
    print(f"Classe : {class_names[idx]}, confiance : {pred[idx]:.2f}")

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python predict_single.py chemin_vers_image.jpg")
    else:
        predict(sys.argv[1])

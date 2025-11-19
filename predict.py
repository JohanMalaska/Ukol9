import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Načtení modelu
model = tf.keras.models.load_model("bridge_classifier.h5")

# Načtení labelů
with open("labels.txt", "r") as f:
    labels = [l.strip() for l in f.readlines()]

def classify_bridge(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    idx = np.argmax(preds)
    confidence = preds[idx]

    return labels[idx], float(confidence)

# Test
if __name__ == "__main__":
    path = input("Zadej cestu k obrázku: ")
    bridge_type, conf = classify_bridge(path)
    print(f"Typ mostu: {bridge_type}")
    print(f"Jistota: {conf:.2f}")

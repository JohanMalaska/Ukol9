import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# ----------------------------------------------
# 1) Předtrénovaný model (EfficientNet) — pozná obecné objekty
# ----------------------------------------------

general_model = tf.keras.applications.EfficientNetB0(weights="imagenet")
general_preprocess = tf.keras.applications.efficientnet.preprocess_input
general_decode = tf.keras.applications.imagenet_utils.decode_predictions

# ----------------------------------------------
# 2) Tvůj vlastní model, který rozlišuje TYPY mostů
# ----------------------------------------------

bridge_model = tf.keras.models.load_model("bridge_classifier.h5")

with open("labels.txt", "r") as f:
    bridge_labels = [l.strip() for l in f.readlines()]


# ----------------------------------------------
# FUNKCE: získá top-1 předpověď předtrénované AI
# ----------------------------------------------

def general_classify(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = general_preprocess(arr)

    preds = general_model.predict(arr)
    decoded = general_decode(preds, top=3)[0]

    # vracíme top-1 (nejpravděpodobnější objekt)
    class_id, label, prob = decoded[0]
    return label.lower(), prob, decoded


# ----------------------------------------------
# FUNKCE: Tvůj model → rozpoznání typu mostu
# ----------------------------------------------

def classify_bridge_type(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = bridge_model.predict(arr)[0]
    idx = np.argmax(preds)
    confidence = preds[idx]

    return bridge_labels[idx], float(confidence)


# ----------------------------------------------
# HLAVNÍ PROGRAM
# ----------------------------------------------

if __name__ == "__main__":
    path = input("Zadej cestu k obrázku: ")

    # 1) Zjistíme, co je na obrázku celkově
    general_label, general_prob, details = general_classify(path)

    print("🔍 Předtrénovaný model říká, že je na obrázku:")
    print(f"➡️  {general_label}  (jistota {general_prob:.2f})")

    # 2) Je to most?
    if "bridge" not in general_label:
        print("\n❌ Toto NENÍ most.")
        print("👁️ Kompletní TOP3 předpověď AI:")
        for (_, label, prob) in details:
            print(f" - {label}: {prob:.2f}")
    else:
        print("\n✔️ AI potvrdila, že na obrázku je MOST.")
        
        # použijeme tvůj model na specifický typ
        btype, conf = classify_bridge_type(path)
        print(f"🧠 Typ mostu: {btype}  (jistota {conf:.2f})")

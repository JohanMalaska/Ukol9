import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# 1) Nastavení datasetu
# -----------------------------------------------------
img_size=(224, 224)
batch_size=32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_gen = datagen.flow_from_directory(
    "dataset",
    target_size=img_size,
    batch_size=batch_size,
    subset="validation",
    class_mode="categorical",
    shuffle=False   # DŮLEŽITÉ: musíme vypnout shuffle!
)

class_names = list(val_gen.class_indices.keys())

# -----------------------------------------------------
# 2) Načtení modelu
# -----------------------------------------------------
model = tf.keras.models.load_model("bridge_classifier.h5")

# -----------------------------------------------------
# 3) Předikce na validačních datech
# -----------------------------------------------------
pred_probs = model.predict(val_gen)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = val_gen.classes

# -----------------------------------------------------
# 4) Výpis klasifikační zprávy – OPRAVENÁ VERZE
# -----------------------------------------------------
labels = list(range(len(class_names)))  # [0, 1, 2, 3, 4]

print("\n============================")
print("📊 Klasifikační report")
print("============================")
print(classification_report(
    true_labels,
    pred_labels,
    labels=labels,
    target_names=class_names,
    zero_division=0
))

# -----------------------------------------------------
# 5) Matice záměn
# -----------------------------------------------------
cm = confusion_matrix(true_labels, pred_labels)

print("\n============================")
print("📌 Matice záměn (číselně)")
print("============================")
print(cm)

# -----------------------------------------------------
# 6) Graf matice záměn
# -----------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predikce")
plt.ylabel("Skutečnost")
plt.title("Matice záměn – validační data")
plt.tight_layout()
plt.show()

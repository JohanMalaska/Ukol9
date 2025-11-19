import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Nastavení
img_size = (224, 224)
batch_size = 32
epochs = 12

# Dataset + augmentace
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    "dataset",
    target_size=img_size,
    batch_size=batch_size,
    subset="training",
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    "dataset",
    target_size=img_size,
    batch_size=batch_size,
    subset="validation",
    class_mode="categorical"
)

# Uložení labelů
labels = list(train_gen.class_indices.keys())
with open("labels.txt", "w") as f:
    for l in labels:
        f.write(l + "\n")

# Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_size + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trénování
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

model.save("bridge_classifier.h5")
print("Model uložen jako bridge_classifier.h5")

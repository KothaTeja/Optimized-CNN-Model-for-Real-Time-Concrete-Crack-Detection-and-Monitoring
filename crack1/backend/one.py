import os

base_dir = "/content/drive/MyDrive/Ext__Face/Extracted Faces"
focused_path = os.path.join(base_dir, "Focused")
not_focused_path = os.path.join(base_dir, "Not Focused")

if not os.path.exists(base_dir):
    raise FileNotFoundError(f"❌ Dataset folder not found: {base_dir}")
else:
    print(f"Dataset folder exists: {base_dir}")

num_focused = len(os.listdir(focused_path)) if os.path.exists(focused_path) else 0
num_not_focused = len(os.listdir(not_focused_path)) if os.path.exists(not_focused_path) else 0
print(f"Dataset Summary: Focused={num_focused}, Not Focused={num_not_focused}")


from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Image Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    validation_split=0.2  # 20% validation
)

batch_size = 16
target_size = (224, 224)

# ✅ Load Training & Validation Data
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)


import tensorflow as tf
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

base_model = NASNetMobile(weights=None, include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)  # Prevent overfitting
output_layer = Dense(1, activation="sigmoid", dtype=tf.float32)(x)  # Binary Classification

model = Model(inputs=base_model.input, outputs=output_layer)

model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

model.fit(train_generator, validation_data=val_generator, epochs=10)

model.save("/content/drive/MyDrive/focused_notfocused_nasnetmobile.h5")
print(" Model saved successfully!")
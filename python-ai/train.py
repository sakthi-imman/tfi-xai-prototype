# train.py
# --------------------------------------------------
# Train ResNet-50 for document tampering detection
# --------------------------------------------------

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# Paths are relative to this file (python-ai/)
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROCESSED   = os.path.join(BASE_DIR, "processed")
TRAIN_DIR   = os.path.join(PROCESSED, "train")
VAL_DIR     = os.path.join(PROCESSED, "val")

MODELS_DIR  = os.path.join(BASE_DIR, "models")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

MODEL_OUT   = os.path.join(MODELS_DIR, "resnet50_tamper.h5")
LOG_FILE    = os.path.join(LOGS_DIR, "training_log.csv")

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS_STAGE1 = 10   # feature extraction
EPOCHS_STAGE2 = 10   # fine-tuning

SEED = 42

# --------------------------------------------------
# SANITY CHECK FOR DIRECTORIES
# --------------------------------------------------
if not (os.path.isdir(TRAIN_DIR) and os.path.isdir(VAL_DIR)):
    raise FileNotFoundError(
        f"Expected directories:\n{TRAIN_DIR}\n{VAL_DIR}\nPlease create processed/train and processed/val with 'genuine' and 'tampered' subfolders."
    )

# --------------------------------------------------
# DATASET LOADING (tf.data)
# --------------------------------------------------

print("Loading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["genuine", "tampered"],  # enforce stable order
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["genuine", "tampered"],
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
print(f"Class mapping: {class_names} (0 = {class_names[0]}, 1 = {class_names[1]})")

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# --------------------------------------------------
# DATA AUGMENTATION LAYERS
# --------------------------------------------------

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),          # mostly harmless for docs
        layers.RandomRotation(0.03),              # small rotation (± ~5°)
        layers.RandomZoom(0.05),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(factor=0.1),
    ],
    name="data_augmentation",
)

# --------------------------------------------------
# BUILD MODEL — RESNET-50 BACKBONE
# --------------------------------------------------

print("Building model...")

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,),
)

# First, freeze the base for stage 1
base_model.trainable = False

# Input & preprocessing
inputs = keras.Input(shape=IMG_SIZE + (3,), name="input_image")

# ResNet50 expects preprocess_input
x = data_augmentation(inputs)
x = tf.keras.applications.resnet50.preprocess_input(x)

x = base_model(x, training=False)          # frozen in stage 1
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid", name="tamper_prob")(x)

model = keras.Model(inputs, outputs, name="resnet50_tamper_detector")

model.summary()

# --------------------------------------------------
# COMPILE — STAGE 1 (FEATURE EXTRACTION)
# --------------------------------------------------

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ],
)

# --------------------------------------------------
# CALLBACKS
# --------------------------------------------------

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

checkpoint = keras.callbacks.ModelCheckpoint(
    MODEL_OUT,
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

csv_logger = keras.callbacks.CSVLogger(LOG_FILE, append=False)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-7,
)

callbacks = [early_stop, checkpoint, csv_logger, reduce_lr]

# --------------------------------------------------
# TRAIN — STAGE 1
# --------------------------------------------------

print("\n========== STAGE 1: Train top classifier with frozen ResNet50 ==========\n")

history_stage1 = model.fit(
    train_ds,
    epochs=EPOCHS_STAGE1,
    validation_data=val_ds,
    callbacks=callbacks,
)

# --------------------------------------------------
# FINE-TUNING — STAGE 2 (UNFREEZE PART OF RESNET)
# --------------------------------------------------

print("\nUnfreezing top ResNet50 layers for fine-tuning...\n")

# Unfreeze last N layers of ResNet50
fine_tune_at = 100  # you can adjust this
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ],
)

print("\n========== STAGE 2: Fine-tuning ==========\n")

history_stage2 = model.fit(
    train_ds,
    epochs=EPOCHS_STAGE2,
    validation_data=val_ds,
    callbacks=callbacks,
)

print("\nTraining complete.")
print(f"Best model saved to: {MODEL_OUT}")
print(f"Training log saved to: {LOG_FILE}")

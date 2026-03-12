"""
MobileNetV2 transfer learning trainer — Phase 2

Loads images from data/valid/ and data/invalid/, applies augmentation,
fine-tunes MobileNetV2 (pretrained on ImageNet), and exports the result
as models/card_model.tflite.

Usage:
    python training/train_model.py

Requirements:
    - At least ~200 images in data/valid/ and data/invalid/ each
    - venv activated with requirements installed
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import albumentations as A
import cv2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VALID_DIR    = os.path.join("data", "valid")
INVALID_DIR  = os.path.join("data", "invalid")
MODEL_DIR    = "models"
TFLITE_PATH  = os.path.join(MODEL_DIR, "card_model.tflite")
H5_PATH      = os.path.join(MODEL_DIR, "card_model.h5")

IMG_SIZE     = 224   # MobileNetV2 expects 224x224
BATCH_SIZE   = 16
EPOCHS       = 20
VAL_SPLIT    = 0.2   # 20% of images held out for validation

# ---------------------------------------------------------------------------
# Augmentation pipeline (albumentations)
# ---------------------------------------------------------------------------
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
])


def load_images(directory, label):
    """Load all .jpg images from a directory, return (images, labels) arrays."""
    images, labels = [], []
    for fname in os.listdir(directory):
        if not fname.lower().endswith(".jpg"):
            continue
        path = os.path.join(directory, fname)
        img  = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(label)
    return images, labels


def apply_augmentation(images, labels, multiplier=3):
    """
    Augment the dataset by applying the pipeline `multiplier` times per image.
    Returns original + augmented images combined.
    """
    aug_images, aug_labels = list(images), list(labels)
    for img, label in zip(images, labels):
        for _ in range(multiplier):
            result = augment(image=img)
            aug_images.append(result["image"])
            aug_labels.append(label)
    return np.array(aug_images), np.array(aug_labels)


def build_model():
    """
    MobileNetV2 with ImageNet weights, top layers replaced for binary classification.
    Base layers are frozen initially; top layers are trained first.
    """
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze base during initial training

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),  # binary: valid (1) / invalid (0)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def fine_tune(model, base, x_train, y_train, x_val, y_val):
    """Unfreeze the top 30 layers of MobileNetV2 and retrain at a lower LR."""
    for layer in base.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=BATCH_SIZE,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    )


def export_tflite(model, path):
    """Convert and save the Keras model as a TFLite flatbuffer."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {path}")


def main():
    # 1. Load images
    print("Loading images...")
    valid_imgs,   valid_labels   = load_images(VALID_DIR,   label=1)
    invalid_imgs, invalid_labels = load_images(INVALID_DIR, label=0)

    print(f"  Valid:   {len(valid_imgs)}")
    print(f"  Invalid: {len(invalid_imgs)}")

    if len(valid_imgs) < 20 or len(invalid_imgs) < 20:
        print("Not enough images — collect more samples first (aim for 200+ each).")
        return

    all_images = np.array(valid_imgs + invalid_imgs, dtype=np.uint8)
    all_labels = np.array(valid_labels + invalid_labels)

    # 2. Augment
    print("Augmenting dataset...")
    all_images, all_labels = apply_augmentation(all_images.tolist(), all_labels.tolist())
    print(f"  Total after augmentation: {len(all_images)}")

    # 3. Normalise to [0, 1]
    all_images = all_images.astype("float32") / 255.0

    # 4. Shuffle and split
    indices = np.random.permutation(len(all_images))
    all_images = all_images[indices]
    all_labels = all_labels[indices]

    split = int(len(all_images) * (1 - VAL_SPLIT))
    x_train, x_val = all_images[:split], all_images[split:]
    y_train, y_val = all_labels[:split], all_labels[split:]

    # 5. Initial training (frozen base)
    print("\nTraining top layers...")
    model, base = build_model()
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(patience=4, restore_best_weights=True),
            ModelCheckpoint(H5_PATH, save_best_only=True),
        ],
    )

    # 6. Fine-tune top layers of base
    print("\nFine-tuning MobileNetV2 top layers...")
    fine_tune(model, base, x_train, y_train, x_val, y_val)

    # 7. Evaluate
    loss, acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"\nValidation accuracy: {acc * 100:.1f}%")
    if acc < 0.85:
        print("Warning: accuracy below 85% target — collect more training data.")

    # 8. Export
    export_tflite(model, TFLITE_PATH)


if __name__ == "__main__":
    main()

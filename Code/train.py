import numpy as np
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import pickle

# Define paths (relative to Code directory)
real_dir = "../real_and_fake_face_detection/real_and_fake_face/training_real/"
fake_dir = "../real_and_fake_face_detection/real_and_fake_face/training_fake/"

# Function to load images from directory
def load_images_from_directory(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            image = cv2.imread(filepath)
            if image is None:
                continue
            image = cv2.resize(image, (96, 96))
            image = image / 255.0  # Normalize
            images.append(image)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    return images, labels

# Load all images
print("Loading real images...")
real_images, real_labels = load_images_from_directory(real_dir, 0)
print(f"Loaded {len(real_images)} real images")

print("Loading fake images...")
fake_images, fake_labels = load_images_from_directory(fake_dir, 1)
print(f"Loaded {len(fake_images)} fake images")

# Combine
X = np.array(real_images + fake_images)
y = np.array(real_labels + fake_labels)

# Shuffle
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

# Split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nDataset split:")
print(f"  Train: {len(X_train)} images")
print(f"  Val:   {len(X_val)} images")
print(f"  Test:  {len(X_test)} images")

# Save test set for evaluation
with open('test_data.pkl', 'wb') as f:
    pickle.dump({'X_test': X_test, 'y_test': y_test}, f)
print("[OK] Test set saved as 'test_data.pkl'")

# Visualize sample training images
fig = plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(X_train[i][:, :, ::-1])  # BGR to RGB
    plt.title("Real" if y_train[i] == 0 else "Fake")
    plt.axis('off')
plt.suptitle("Sample Training Images", fontsize=20)
plt.tight_layout()
plt.show()

# Data augmentation
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1)

# MobileNetV2 model
print("\nBuilding MobileNetV2 model...")
mnet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
tf.keras.backend.clear_session()

model = Sequential([
    mnet,
    GlobalAveragePooling2D(),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.1),
    Dense(2, activation="softmax")
])

# Freeze base model weights
model.layers[0].trainable = False

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# Learning rate scheduler and early stopping
def scheduler(epoch):
    if epoch <= 2:
        return 0.001
    elif epoch > 2 and epoch <= 15:
        return 0.0001
    else:
        return 0.00001

lr_scheduler = LearningRateScheduler(scheduler)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train with data augmentation
print("\nTraining model...")
hist = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=20,
    validation_data=(X_val, y_val),
    callbacks=[lr_scheduler, early_stop],
    steps_per_epoch=len(X_train)//32)

# Save model
model.save('deepfake_detection_model.h5')
print("[OK] Model saved as 'deepfake_detection_model.h5'")

# Visualizing accuracy and loss
epochs_range = range(len(hist.history['loss']))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, hist.history['loss'], label='Train Loss', marker='o')
plt.plot(epochs_range, hist.history['val_loss'], label='Val Loss', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, hist.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(epochs_range, hist.history['val_accuracy'], label='Val Accuracy', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs Validation Accuracy')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# Calculate training and validation metrics
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"{'='*50}")
print("TRAINING SUMMARY")
print(f"{'='*50}")
print(f"Final Train Accuracy:       {train_acc:.4f}")
print(f"Final Validation Accuracy:  {val_acc:.4f}")
print(f"Test Accuracy (held-out):   {test_acc:.4f}")
print(f"\nFinal Train Loss:           {train_loss:.4f}")
print(f"Final Validation Loss:      {val_loss:.4f}")
print(f"Test Loss (held-out):       {test_loss:.4f}")
print(f"{'='*50}")
print("\n[OK] Run 'evaluate.py' for detailed metrics (confusion matrix, F1, ROC-AUC)")


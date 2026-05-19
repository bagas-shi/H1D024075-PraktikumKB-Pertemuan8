# ==========================================
# 1. IMPORT LIBRARY (Langkah 3 & 4)
# ==========================================
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# 2. PREPROCESSING DATA DENGAN IMAGEDATAGENERATOR (Langkah 4)
# ==========================================
# Path menuju folder dataset utama
dataset_path = "./rockpaperscissors"

# Inisialisasi generator dengan normalisasi piksel dan pembagian data validasi (20%)
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Mengubah nilai piksel dari [0-255] menjadi [0-1]
    validation_split=0.2     # Memisahkan 20% data untuk validasi
)

# Generator untuk data latihan (Training)
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),  # Mengubah ukuran gambar menjadi 150x150
    batch_size=32,           # Ukuran batch proses harian
    class_mode='categorical',# Klasifikasi multi-kelas (rock, paper, scissors)
    subset='training'        # Mengambil subset data training
)

# Generator untuk data validasi (Validation)
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'      # Mengambil subset data validasi
)

# ==========================================
# 3. MEMBANGUN ARSITEKTUR MODEL CNN (Langkah 5)
# ==========================================
model = Sequential([
    # Layer Konvolusi ke-1 + Max Pooling
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    
    # Layer Konvolusi ke-2 + Max Pooling
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Layer Konvolusi ke-3 + Max Pooling
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Flattening data untuk masuk ke Fully Connected Layer (MLP)
    Flatten(),
    
    # Hidden Layer (Dense) dengan 512 neuron
    Dense(512, activation='relu'),
    
    # Output Layer dengan 3 neuron sesuai jumlah kelas (Rock, Paper, Scissors)
    Dense(3, activation='softmax')
])

# Menampilkan ringkasan/arsitektur model di konsol
model.summary()

# ==========================================
# 4. KOMPILASI MODEL (Langkah 6)
# ==========================================
model.compile(
    loss='categorical_crossentropy', # Sesuai petunjuk untuk klasifikasi multi-kelas
    optimizer='adam',                # Menggunakan Adam optimizer
    metrics=['accuracy']
)

# ==========================================
# 5. PELATIHAN MODEL / MODEL FITTING (Langkah 7)
# ==========================================
print("\n--- Memulai Proses Pelatihan Model ---")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10 # Model dilatih sebanyak 10 epoch
)

# ==========================================
# 6. EVALUASI MODEL (Langkah 8 & 9)
# ==========================================
print("\n--- Mengevaluasi Model pada Data Validasi ---")
val_loss, val_acc = model.evaluate(validation_generator)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}\n')

# Mencoba melakukan prediksi pada data validasi
print("--- Menampilkan Contoh Prediksi (Probabilitas Tiap Kelas) ---")
predictions = model.predict(validation_generator)
print(predictions)
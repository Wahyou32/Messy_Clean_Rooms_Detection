"""
Improved Room Classification Model (Messy vs Clean)
Features:
- Transfer Learning with MobileNetV2 for better accuracy
- Advanced data augmentation to prevent overfitting
- Batch normalization for faster convergence
- Dropout layers for regularization
- Learning rate scheduling
- Early stopping to prevent overfitting
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import zipfile
import absl.logging

# Suppress verbose logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuration
IMG_SIZE = 224  # MobileNetV2 expects 224x224
BATCH_SIZE = 32
EPOCHS = 30
TRAIN_DIR = 'messy-vs-clean-room'

def extract_dataset(zip_path='messy-vs-clean-room.zip'):
    """Extract the dataset from zip file"""
    print(f"Extracting {zip_path}...")
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall('/tmp')
    zip_ref.close()
    return '/tmp/images'

def create_data_generators(base_dir):
    """Create augmented training and validation data generators"""
    
    # Advanced data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,  # Rooms are rarely upside down
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    # Only rescaling for validation/test
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    print("Creating training generator...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        seed=42
    )
    
    print("Creating validation generator...")
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
        seed=42
    )
    
    return train_generator, val_generator

def create_model():
    """Create improved model using transfer learning"""
    
    # Load pre-trained MobileNetV2 (lightweight and efficient)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Build the complete model
    model = tf.keras.Sequential([
        base_model,
        BatchNormalization(),
        Dense(256, activation='relu', kernel_regularizer='l2'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(128, activation='relu', kernel_regularizer='l2'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    return model, base_model

def train_model(model, base_model, train_gen, val_gen):
    """Train the model with callbacks"""
    
    # Compile model with initial learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print("\n=== Phase 1: Training frozen base model ===\n")
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        'best_model_phase1.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Train with frozen base
    history_phase1 = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=10,
        validation_data=val_gen,
        validation_steps=len(val_gen),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=2
    )
    
    # Unfreeze some layers for fine-tuning
    print("\n=== Phase 2: Fine-tuning ===\n")
    base_model.trainable = True
    
    # Freeze most layers, unfreeze last 50 layers
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Fine-tune
    history_phase2 = model.fit(
        train_gen,
        steps_per_epoch=len(train_gen),
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=len(val_gen),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=2
    )
    
    return model, history_phase1, history_phase2

def main():
    """Main training pipeline"""
    print("="*60)
    print("IMPROVED ROOM CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    # Extract dataset
    base_dir = extract_dataset()
    
    # Create data generators
    train_gen, val_gen = create_data_generators(base_dir)
    
    # Print class indices
    print(f"\nClass indices: {train_gen.class_indices}")
    
    # Create model
    print("\nBuilding model...")
    model, base_model = create_model()
    model.summary()
    
    # Train model
    model, hist1, hist2 = train_model(model, base_model, train_gen, val_gen)
    
    # Evaluate on validation set
    print("\n=== Final Evaluation ===")
    val_results = model.evaluate(val_gen, verbose=1)
    print(f"Validation Loss: {val_results[0]:.4f}")
    print(f"Validation Accuracy: {val_results[1]:.4f}")
    print(f"Validation Precision: {val_results[2]:.4f}")
    print(f"Validation Recall: {val_results[3]:.4f}")
    
    # Save the final model
    export_path = 'trained_model_improved'
    print(f'\nSaving model to {export_path}...')
    
    model.save(export_path, overwrite=True, include_optimizer=True)
    
    # Also save in TensorFlow SavedModel format
    print(f'Saving model in SavedModel format...')
    tf.saved_model.save(model, 'saved_model_improved')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Model saved to: {export_path}")
    print(f"SavedModel saved to: saved_model_improved")
    
    return model

if __name__ == '__main__':
    model = main()
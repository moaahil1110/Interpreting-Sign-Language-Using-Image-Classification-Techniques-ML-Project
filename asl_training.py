
"""
ASL Alphabet CNN Model Training Script
Complete implementation for training a CNN model on ASL alphabet dataset
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ASLTrainer:
    def __init__(self, data_path, img_size=(200, 200), batch_size=32):
        """
        Initialize ASL Trainer

        Args:
            data_path: Path to ASL alphabet dataset
            img_size: Target image size (height, width)
            batch_size: Training batch size
        """
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None

        # ASL alphabet classes (excluding J and Z for static gesture recognition)
        self.classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        self.num_classes = len(self.classes)

        print(f"Initialized ASL Trainer with {self.num_classes} classes")
        print(f"Classes: {self.classes}")

    def setup_data_generators(self, validation_split=0.2):
        """
        Setup data generators with augmentation

        Args:
            validation_split: Fraction of data to use for validation
        """
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )

        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )

        # Create training generator
        self.train_generator = train_datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        # Create validation generator
        self.validation_generator = val_datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.validation_generator.samples}")
        print(f"Classes found: {list(self.train_generator.class_indices.keys())}")

    def build_cnn_model(self):
        """
        Build CNN model architecture optimized for ASL recognition
        """
        model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),

            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def build_transfer_learning_model(self, base_model_name='ResNet50'):
        """
        Build transfer learning model using pre-trained base

        Args:
            base_model_name: Name of base model ('ResNet50', 'VGG16', 'InceptionV3')
        """
        # Get base model
        if base_model_name == 'ResNet50':
            base_model = keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif base_model_name == 'VGG16':
            base_model = keras.applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif base_model_name == 'InceptionV3':
            base_model = keras.applications.InceptionV3(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")

        # Freeze base model layers
        base_model.trainable = False

        # Add custom head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print(f"Built transfer learning model with {base_model_name} base")
        return model

    def train_model(self, epochs=50, model_save_path='asl_model.h5'):
        """
        Train the model with callbacks

        Args:
            epochs: Number of training epochs
            model_save_path: Path to save best model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_cnn_model() or build_transfer_learning_model() first")

        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Calculate steps per epoch
        steps_per_epoch = self.train_generator.samples // self.batch_size
        validation_steps = self.validation_generator.samples // self.batch_size

        print(f"Training for {epochs} epochs")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")

        # Train model
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        print(f"Training completed. Best model saved to {model_save_path}")
        return self.history

    def plot_training_history(self):
        """
        Plot training history graphs
        """
        if self.history is None:
            print("No training history available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_model(self):
        """
        Evaluate model on validation data and show detailed metrics
        """
        if self.model is None:
            print("No model available for evaluation")
            return

        # Get predictions
        predictions = self.model.predict(self.validation_generator)
        predicted_classes = np.argmax(predictions, axis=1)

        # Get true labels
        true_classes = self.validation_generator.classes
        class_labels = list(self.validation_generator.class_indices.keys())

        # Classification report
        print("Classification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                  target_names=class_labels))

        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main training function
    """
    # Configuration
    DATA_PATH = "/home/worm/A/ML/mlproj/Dataset/asl_alphabet_train/asl_alphabet_train"  # Path to your dataset
    IMG_SIZE = (200, 200)
    BATCH_SIZE = 32
    EPOCHS = 50

    # Initialize trainer
    trainer = ASLTrainer(DATA_PATH, IMG_SIZE, BATCH_SIZE)

    # Setup data generators
    trainer.setup_data_generators(validation_split=0.2)

    # Build model (choose one)
    # Option 1: CNN from scratch
    trainer.build_cnn_model()

    # Option 2: Transfer learning (uncomment to use)
    # trainer.build_transfer_learning_model('ResNet50')

    # Print model architecture
    trainer.model.summary()

    # Train model
    trainer.train_model(epochs=EPOCHS, model_save_path='best_asl_model.h5')

    # Plot training history
    trainer.plot_training_history()

    # Evaluate model
    trainer.evaluate_model()

    print("Training completed successfully!")

if __name__ == "__main__":
    main()

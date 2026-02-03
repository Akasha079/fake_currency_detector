import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from app.model_definitions import build_mobilenet_model, build_resnet_model, build_custom_cnn

def train(data_dir, model_type='mobilenet', epochs=5, batch_size=32):
    train_dir = os.path.join(data_dir, 'Train')
    test_dir = os.path.join(data_dir, 'Test')

    # Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    print(f"Loading data from {train_dir} and {test_dir}...")
    
    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("Error: Train or Test directories not found in data folder.")
        return

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    num_classes = len(train_generator.class_indices)
    print(f"Detected {num_classes} classes: {list(train_generator.class_indices.keys())}")

    # Build Model
    if model_type == 'mobilenet':
        model = build_mobilenet_model(num_classes=num_classes)
    elif model_type == 'resnet':
        model = build_resnet_model(num_classes=num_classes)
    else:
        model = build_custom_cnn(num_classes=num_classes)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )

    # Save Model
    if not os.path.exists('model'):
        os.makedirs('model')
    model_path = f'model/{model_type}_model.h5'
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy - {model_type}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(f'model/{model_type}_accuracy.png')
    print(f"Accuracy plot saved to model/{model_type}_accuracy.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Fake Currency Detector Model')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory containing Train and Test folders')
    parser.add_argument('--model', type=str, default='mobilenet', choices=['mobilenet', 'resnet', 'cnn'], help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    args = parser.parse_args()

    train(args.data, args.model, args.epochs)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

dataset_dir = 'F:/Projects/AI/Electro AI/Datasets/archive/images'
train_dir = 'F:/Projects/AI/Electro AI/Datasets/archive/images/train'
validation_dir = 'F:/Projects/AI/Electro AI/Datasets/archive/images/validation'


# Set up data generators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='sparse')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='sparse')

# Define your model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(train_generator.class_indices), activation='softmax')  # Adjust the number of classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size)

# Save the model
model_path = os.path.join(dataset_dir, 'model.h5')
model.save(model_path)
print(f"Model saved to {model_path}")

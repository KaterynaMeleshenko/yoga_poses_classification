import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as ts
from keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score


#Load the data
train_df = pd.read_csv("train.csv")
print(train_df.head())

# Count the occurrences of each class
class_counts = train_df["class_6"].value_counts()

# Plot a bar chart of class distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Display a few sample images
sample_images = train_df.sample(n=4)
plt.figure(figsize=(12, 8))
for i, (_, row) in enumerate(sample_images.iterrows()):
    img_path = f"images/train_images/{row['image_id']}"
    img = plt.imread(img_path)
    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plt.title(row['class_6'])
    plt.axis('off')
plt.tight_layout()
plt.show()

# NOrmalization
# Path to the folder containing the training images
train_images_folder = "images/train_images/"

# Initialize empty lists to store images and labels
images = []
labels = []

# Define a target image size for resizing
target_size = (150, 150)

# Loop through the DataFrame rows
for index, row in train_df.iterrows():
    img_filename = row['image_id']
    img_path = os.path.join(train_images_folder, img_filename)
    
    # Load and preprocess the image using TensorFlow's functions
    img = load_img(img_path, target_size=target_size) # Resize to a common size
    img_array = img_to_array(img) 
    img_array /= 255.0  # Normalize pixel values
    
    images.append(img_array)
    
    label = row['class_6']
    labels.append(label)

# Convert lists to arrays
images = np.array(images)
labels = np.array(labels)

### Split the data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)
print("Validation images shape:", val_images.shape)
print("Validation labels shape:", val_labels.shape)

### Model building
import tensorflow as tf

model = tf.keras.models.Sequential([
    # Input layer
    tf.keras.layers.Input(shape=(150, 150, 3)),
    
    # Convolutional layers with Batch Normalization
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    
    # Fully connected layers
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    
    # Output layer
    tf.keras.layers.Dense(6, activation='softmax')
])

# Print the model summary
model.summary()

# Define F1 measures: F1 = 2 * (precision * recall) / (precision + recall)
def custom_f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives+K.epsilon())
        return recall


    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives+K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# Set the training parameters
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=[custom_f1, 'accuracy'])

train_labels_encoded = to_categorical(train_labels, num_classes=6)
val_labels_encoded = to_categorical(val_labels, num_classes=6)

# Create an ImageDataGenerator for training data
training_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create a data generator using flow
train_generator = training_datagen.flow(
    train_images, train_labels_encoded,
    batch_size=32,
)

# Create an ImageDataGenerator for validation data
validation_datagen = ImageDataGenerator()

# Create a generator using flow
validation_generator = validation_datagen.flow(
    val_images, val_labels_encoded,
    batch_size=32, 
    shuffle=False
)

### Model training

# Define a learning rate schedule
def lr_schedule(epoch, lr):
    initial_learning_rate = 0.001
    decay_steps = 10000
    decay_rate = 0.9
    lr = initial_learning_rate * decay_rate**(epoch / decay_steps)
    return lr

callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Train the model
history = model.fit(train_generator, epochs=50, steps_per_epoch=50, validation_data = validation_generator, callbacks=[callback], verbose = 1)

# Plot the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()

### Model evaluation
# Get predictions from the model
val_predictions = model.predict(val_images)
val_pred_labels = np.argmax(val_predictions, axis=1)  # Convert softmax probabilities to class labels

# Calculate micro-averaged F1-Score
micro_f1_score = f1_score(val_labels, val_pred_labels, average='micro')

print("Micro-Averaged F1-Score:", micro_f1_score)

### Predictions

# Path to the test images directory
test_images_dir = 'images/test_images/'

# List all image files in the directory
test_image_files = os.listdir(test_images_dir)

# Initialize an empty list to store preprocessed test images
preprocessed_test_images = []

# Initialize an empty list to store image IDs (assuming the filenames are IDs)
test_image_ids = []

# Create an ImageDataGenerator for test data (no data augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Loop through each image file
for image_file in test_image_files:
    # Construct the image path
    image_path = os.path.join(test_images_dir, image_file)

    # Load and preprocess the image using the Keras load_img function
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    preprocessed_test_images.append(image_array)

    test_image_ids.append(image_file)

# Convert the list of images to a NumPy array
preprocessed_test_images = np.array(preprocessed_test_images)

# Make predictions using the trained model
test_predictions = model.predict(preprocessed_test_images)
test_pred_labels = np.argmax(test_predictions, axis=1)

### Create submission file

submission_df = pd.DataFrame({'image_id': test_image_ids, 'class_6': test_pred_labels})
submission_df.to_csv('submission.csv', index=False)
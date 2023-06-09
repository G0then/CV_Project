import os
import numpy as np
import pandas as pd
import random
import cv2
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
import psutil

# Define the fixed image size for input into the VGG CNN
IMG_SIZE = (224, 224)

# Set the path to your dataset and text file
dataset_path = 'F:/Toolkit/ComputacaoVisual/Images_with_CC/bi_concepts1553/'
dataset_anps = 'flickr_final_dataset_summary.csv'

# Load form CSVs file
df_anps = pd.read_csv(dataset_anps)


# Convert sentiment regression values (-2 to 2) in sentiment classification problem
def get_sentiment_label(sentiment_percentage):
    if sentiment_percentage >= 0.5: # Positive sentiment
        return [0,0,1]
    elif sentiment_percentage <= -0.5: # Negative sentiment
        return [1,0,0]
    else: # Neutral sentiment
        return [0,1,0]


# create an Empty DataFrame object
df = pd.DataFrame(columns = ['File Path', 'Sentiment'])

# Loop through ANPs
for i, row in df_anps.iterrows():
    print("A verificar o ANP: ", row["Folder Name"])
    anp_folder_path = dataset_path + row["Folder Name"] + '/'
    anp_sentiment = get_sentiment_label(row["Sentiment"])

    # Check if the folder exists (because some folder doesnt exist)
    if not os.path.exists(anp_folder_path):
        print(f"Folder does not exist: {anp_folder_path}")
        continue

    # Iterate over image files
    for file_name in os.listdir(anp_folder_path):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(anp_folder_path, file_name)
            # Insert Dict to the dataframe
            new_row = {'File Path': file_path, 'Sentiment': anp_sentiment}
            df.loc[len(df)] = new_row

# Shuffle files
df = shuffle(df)

print("Df Shape: ", df.shape)

# Delete df_anps from memory
del df_anps

# Preprocess the images
def preprocess_image(filename):
    try:
        # Load the image file
        img = cv2.imread(filename)

        # To make sure
        if img is None:
            print(f"Failed to read image: {filename}")
            return None

        # Resize the image to the fixed size
        resized_img = cv2.resize(img, IMG_SIZE)

        # Normalize pixel values to be between 0 and 1
        normalized_img = resized_img.astype('float32') / 255.

        # Check if the image shape matches the expected shape
        if normalized_img.shape != (224, 224, 3):
            return None

        # Transpose the image to match the input format expected by the VGG CNN
        # input_img = np.transpose(normalized_img, (2, 0, 1))
        return normalized_img
    except Exception as e:
        print(f"Error preprocessing image: {filename}")
        print(str(e))
        return None

def plot_training_curves(history, save_path=None):
    # Get the training and validation loss values
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get the training and validation RMSE values
    #rmse = history.history['root_mean_squared_error']
    #val_rmse = history.history['val_root_mean_squared_error']

    # Get the training and validation accuracy values
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Plot the training and validation loss curves
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    #plt.plot(epochs, rmse, 'g', label='Training RMSE')
    #plt.plot(epochs, val_rmse, 'm', label='Validation RMSE')
    plt.plot(epochs, accuracy, 'g', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'm', label='Validation Accuracy')
    plt.title('Training and Validation Curves')
    plt.xlabel('Epochs')
    #plt.ylabel('Loss/RMSE')
    plt.ylabel('Loss/Accuracy')
    plt.legend()

    # Save the plot to a file if save_path is provided
    if save_path is not None:
        plt.savefig(save_path)

    #plt.show()

##### Base Model #####
# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Add a new fully connected layers to the model for classification
x = base_model.output
#x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
#x = Dense(2048, activation='relu')(x)
#x = Dense(1024, activation='relu')(x)
x = Dense(24, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x) # For Classification
#predictions = Dense(1, activation='linear')(x) # For Regression
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers in the base model, so it prevents that pre-trained VGG16 CNN model weights from being updated during the training process
for layer in base_model.layers[:-8]:
    layer.trainable = False

#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

# Compile
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']) # For Classification
#model.compile(optimizer=optimizer, loss='mse', metrics=['RootMeanSquaredError']) # For Regression

# Split data into train, validation and test data
def split_data(X, y):
    # Convert arrays to numpy arrays
    X = np.array(X)
    y = np.array(y)

    print("X length: ", len(X), "   Y length: ", len(y))

    # Create training, validation and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Transpose input data
    # X_train = np.transpose(X_train, (0, 2, 3, 1))
    # X_val = np.transpose(X_val, (0, 2, 3, 1))
    # X_test = np.transpose(X_test, (0, 2, 3, 1))

    # Monitorizar os dados aqui
    print("Monitorizar os dados:")
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)
    print("X_val: ", X_val.shape)
    print("y_val: ", y_val.shape)
    print("X_test: ", X_test.shape)
    print("y_test: ", y_test.shape)

    return (X_train, y_train, X_val, y_val, X_test, y_test)


# Train and save the model
def train_and_save_model(data, model_path):
    (X_train, y_train, X_val, y_val, X_test, y_test) = data

    # Fit model
    history = model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_val, y_val))
    plot_training_curves(history, save_path='training_curves.png')

    # Evaluate the performance of the model using test dataset
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    # Save model
    model.save(model_path)


# Set a threshold for memory usage
memory_threshold = 6000  # Specify the memory threshold in MB
n_files_threshold = 100000  # Specify the threshold of number of files
file_count = 0

# Load the image data and labels into numpy arrays
X = []
y = []
data = []

# Model Path
model_path = "sentiment_model.h5"

for i, row in df.iterrows():
    print("A verificar o ANP: ", row["File Path"])

    image_preprocessed = preprocess_image(row["File Path"])
    # To make sure
    if image_preprocessed is not None:
        X.append(image_preprocessed)
        y.append(row["Sentiment"])
        file_count += 1
    else:
        continue

    # Check memory consumption
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # Memory usage in MB
    print(f"Files {file_count} - Current memory usage: {memory_usage} MB")

    # Check number of files or memory threshold was exceeded
    if memory_usage > memory_threshold:
        # if file_count >= n_files_threshold or (i == (len(shuffled_anp_sentiment_mapping) - 1) and len(X) >= 2000):
        data = split_data(X, y)
        train_and_save_model(data, model_path)

        # Clear X and y to release memory
        X = []
        y = []
        data = []
        # Reset counter
        file_count = 0

# Train and save the final model
#model_path = "sentiment_model.h5"
#train_and_save_model(X, y, model_path)
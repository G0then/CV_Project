import os
import numpy as np
import pandas as pd
import cv2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split

# Define the fixed image size for input into the VGG CNN
IMG_SIZE = (224, 224)

# Load the data into a pandas dataframe
tweet_data = pd.read_csv('F:/Toolkit/ComputacaoVisual/t4sa_text_sentiment.tsv', delimiter='\t')

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
        normalized_img = resized_img.astype('float32') / 255.0

        # Transpose the image to match the input format expected by the VGG CNN
        # input_img = np.transpose(normalized_img, (2, 0, 1))
        return normalized_img
    except Exception as e:
        print(f"Error preprocessing image: {filename}")
        print(str(e))
        return None

# Load the image data and labels into numpy arrays
X = []
y = []
for tweet_id in tweet_data['TWID']:
    print("A verificar o tweet id: ", tweet_id)
    folder_path = 'F:/Toolkit/ComputacaoVisual/b-t4sa_imgs/data/'+str(tweet_id)[0:5]+'/'

    # Check if the folder exists (because some folder doesnt exist)
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        continue

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if str(tweet_id) in file_name and ("person" not in file_name) and os.path.isfile(file_path):
            X.append(preprocess_image(file_path))
            y.append([tweet_data.loc[tweet_data['TWID']==tweet_id, 'NEG'].values[0],
                      tweet_data.loc[tweet_data['TWID']==tweet_id, 'NEU'].values[0],
                      tweet_data.loc[tweet_data['TWID']==tweet_id, 'POS'].values[0]])
X = np.array(X)
y = np.array(y)

# Create training, validation and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Add a new fully connected layers to the model for classification
x = base_model.output
#x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(2048, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(24, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers in the base model, so it prevents that pre-trained VGG16 CNN model weights from being updated during the training process
for layer in base_model.layers:
    layer.trainable = False

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Transpose input data
#X_train = np.transpose(X_train, (0, 2, 3, 1))
#X_val = np.transpose(X_val, (0, 2, 3, 1))
#X_test = np.transpose(X_test, (0, 2, 3, 1))

# Fit model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the performance of the model using test dataset
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Save model
model.save('sentiment_model.h5')

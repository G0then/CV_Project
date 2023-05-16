import cv2
import numpy as np
import os
import tensorflow as tf
import LogisticRegression as LG
import Lion as LION
import InceptionV3 as ICP
import InceptionResNet as IRN
from histogram import generate_histograms

# Define the classes you want to classify your images into
# classes = ['forest', 'beach', 'kitchen']

TRAIN_DIR = 'images/seg_train/seg_train'
TEST_DIR = 'images/seg_test/seg_test'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    seed=123,
    image_size=(150, 150),
    batch_size=64,
    label_mode = 'categorical')

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    seed=123,
    image_size=(150, 150),
    batch_size=64,
    label_mode = 'categorical')
print(train_ds)

class_names = train_ds.class_names
print(class_names)

histograms = generate_histograms(train_ds, class_names)
print(histograms)

# # Load your dataset and extract pixel values and class labels
# data = []
# labels = []
# images = []
# for c in classes:
#     for file in os.listdir('images/' + c):
#         img = cv2.imread('images/' + c + '/' + file)
#         # img = cv2.resize(img, (224, 224))
#         img = cv2.resize(img, (64, 64))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         hist = cv2.calcHist([img], [0, 1], None, [256, 256], [0, 256, 0, 256])
#         hist = cv2.normalize(hist, hist).flatten()
#         images.append(img)
#         data.append(hist)
#         labels.append(classes.index(c))

# # Convert data and labels to numpy arrays
# images = np.array(images)
# data = np.array(data)
# labels = np.array(labels)

# Train a classifier on the data and labels (e.g. using scikit-learn)
# ICP.CategorizeImage(train_ds,val_ds,class_names)
IRN.CategorizeImage(train_ds,val_ds,class_names)
# LION.CategorizeImage(train_ds,val_ds,class_names)

# Generate color histograms for each class
# class_hists = {}
# for c in classes:
#     class_data = data[labels == classes.index(c)]
#     class_hist = np.zeros((256, 256))
#     for hist in class_data:
#         class_hist += hist.reshape(256, 256)
#     class_hists[c] = cv2.normalize(class_hist, class_hist).flatten()

# Classify new images using your trained classifier and the ideal color histograms
def classify_image(classifier):
    # Preprocess the image
    img = cv2.imread('images/glacier.jpg')
    img = cv2.resize(img, (150, 150))

    hist = cv2.calcHist([img], [0, 1], None, [256, 256], [0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    # Predict the class of the image using your classifier
    pred_class = classifier.predict(hist.reshape(1, -1))[0]
    # Use the ideal color histogram for the predicted class to extract color features
    ideal_hist = class_hists[classes[pred_class]]
    color_features = np.hstack((hist, ideal_hist))
    return color_features
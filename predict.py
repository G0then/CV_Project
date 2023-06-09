import os
import numpy as np
import cv2
from keras.models import load_model

# Define the fixed image size for input into the VGG CNN
IMG_SIZE = (224, 224)

# Load the trained model
model = load_model('sentiment_model.h5')

#Treshold
sentiment_threshold = 0.7

# Preprocess image for input into the model
def preprocess_image(filename):
    # Load the image file
    img = cv2.imread(filename)
    # Resize the image to the fixed size
    resized_img = cv2.resize(img, IMG_SIZE)
    # Normalize pixel values to be between 0 and 1
    normalized_img = resized_img.astype('float32') / 255.0
    # Transpose the image to match the input format expected by the VGG CNN
    #input_img = np.transpose(normalized_img, (2, 0, 1))
    return normalized_img

# Predict image sentiment
def predict_sentiment(filename):
    # Preprocess the image for input into the model
    input_img = preprocess_image(filename)
    # Add an extra dimension to the image array to match the input shape expected by the model
    input_img = np.expand_dims(input_img, axis=0)
    # Make a prediction with the imported model created in "trainModel.py"
    prediction = model.predict(input_img)[0] # For Classification
    #prediction = model.predict(input_img) # For Regression
    # Create a dictionary of sentiment percentages
    sentiment_percentages = {
        'negative': prediction[0],
        'neutral': prediction[1],
        'positive': prediction[2]
    }
    # Return the sentiment percentages dictionary
    return sentiment_percentages

# Convert sentiment values (-2 to 2) in sentiment classification labels
def get_sentiment_label(sentiment_percentage):
    if sentiment_percentage >= 0.5:
        return "Positive"
    elif sentiment_percentage <= -0.5:
        return "Negative"
    else:
        return "Neutral"

# Predict
filename = 'images/sky2.jpg'
sentiment_percentages = predict_sentiment(filename)
#sentiment_label = get_sentiment_label(sentiment_percentages)
print(sentiment_percentages)

# Best sentiment predicted
max_sentiment = max(sentiment_percentages, key=sentiment_percentages.get)

# Verify is best sentiment predicted value is higher than threshold
if sentiment_percentages[max_sentiment] >= sentiment_threshold:
    print("Sentiment " + max_sentiment + " detected in image!")


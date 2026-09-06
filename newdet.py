import tensorflow as tf
import numpy as np
from PIL import Image

# Load expression classification model
model = tf.keras.models.load_model('emtion_cnn_model.h5')

# Use with any face detector
def predict_expression(face_image):
    predictions = model.predict(face_image)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    return dict(zip(emotions, predictions[0]))
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('emotion_cnn_model.h5')
print("Model loaded successfully.")

# Create a dummy input with the expected shape
dummy_input = np.random.rand(1, 48, 48, 1).astype(np.float32)

# Test prediction
preds = model.predict(dummy_input)
print("Prediction successful:", preds)
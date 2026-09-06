import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('emotion_cnn_model.h5')

# Define emotion labels (update if your dataset uses different mapping)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Convert to grayscale and resize to 48x48
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.reshape(1, 48, 48, 1).astype(np.float32) / 255.0
    # Predict emotion
    preds = model.predict(face)
    emotion = emotion_labels[np.argmax(preds)]
    # Display result
    cv2.putText(frame, f'Emotion: {emotion}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Live Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

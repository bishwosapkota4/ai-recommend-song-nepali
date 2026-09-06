import pandas as pd
import numpy as np
import cv2
from fer.fer import FER

CSV_PATH = 'archive/ckextended.csv'  # Path to your CSV file
OUTPUT_FILE = 'csv_emotion_results.txt'

# Read the CSV
df = pd.read_csv(CSV_PATH)

# Initialize FER detector
fer_detector = FER(mtcnn=False)

results = []

for idx, row in df.iterrows():
    pixels = np.fromstring(row['pixels'], sep=' ')
    img_size = int(np.sqrt(len(pixels)))
    img = pixels.reshape((img_size, img_size)).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    emotions = fer_detector.detect_emotions(img_bgr)
    results.append((idx, row['emotion'], emotions))
    print(f"Row {idx} | Label: {row['emotion']} | Detected: {emotions}")

# Optionally, save results to a file
with open(OUTPUT_FILE, 'w') as f:
    for idx, label, emotions in results:
        f.write(f"Row {idx} | Label: {label} | Detected: {emotions}\n")

print(f"Results saved to {OUTPUT_FILE}")

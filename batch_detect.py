import os
import cv2
from fer.fer import FER

# Set the path to your dataset folder
DATASET_DIR = 'your_dataset_folder'  # Change this to your folder path
OUTPUT_FILE = 'emotion_results.txt'

# Initialize FER detector
fer_detector = FER(mtcnn=False)

results = []

for filename in os.listdir(DATASET_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(DATASET_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}")
            continue
        emotions = fer_detector.detect_emotions(img)
        results.append((filename, emotions))
        print(f"{filename}: {emotions}")

# Optionally, save results to a file
with open(OUTPUT_FILE, 'w') as f:
    for filename, emotions in results:
        f.write(f"{filename}: {emotions}\n")

print(f"Results saved to {OUTPUT_FILE}")

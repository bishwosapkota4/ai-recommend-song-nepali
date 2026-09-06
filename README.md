# AI Emotion-Based Song Recommendation

An experimental Nepali music recommendation project that combines facial-emotion recognition with content-based song similarity. The project includes a Flask API, a CNN emotion classifier, lyric and audio feature matrices, and standalone scripts for testing and dataset experiments.

## What It Does

The main application provides two related capabilities:

1. **Emotion detection from a webcam**: captures a short video, detects the largest face with OpenCV, classifies each face crop with a Keras CNN, and returns the dominant emotion with an average confidence score.
2. **Song recommendations**: recommends songs similar to a user's liked songs using one of three similarity modes:
   - `overall`: TF-IDF similarity from artist, genre, and mood
   - `lyrics`: similarity from the precomputed lyric feature vectors
   - `audio`: similarity from the precomputed MFCC vectors

The emotion model recognizes `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, and `Neutral`. The current Flask service detects emotion but does not automatically pass that emotion into the recommendation endpoints; a client can use the detected result to choose its own recommendation flow.

## Project Structure

| File | Purpose |
| --- | --- |
| `app.py` | Main Flask API. Loads the song data, builds similarity matrices, exposes recommendation and webcam emotion endpoints, and runs on port `5001`. |
| `recommend_songs.py` | Command-line recommender that reads `combined_similarity_matrix.csv`. |
| `train_emotion_model.py` | Trains the 48x48 grayscale CNN using `fer2013/train` and `fer2013/test`. Saves the best and final H5 models. |
| `live_emotion_predict.py` | Simple webcam emotion prediction loop using the CNN model. Press `Q` to exit. |
| `predict.py` | Loads the CNN and performs a dummy-input prediction smoke test. |
| `newdet.py` | Standalone real-time FER detector that draws the top emotions on camera frames. |
| `batch_detect.py` | FER experiment for detecting emotions in images from a configurable directory. |
| `detect_from_csv.py` | FER experiment for reading pixel rows from `archive/ckextended.csv`. |
| `test1.py` | Small FER real-time processing experiment that processes every third frame. |
| `musicdata.csv` | Song catalogue with artist, title, genre, mood, lyric path, and audio-feature path metadata. |
| `lyric_features.csv` | Precomputed lyric word-feature vectors keyed by `song_title`. |
| `audio_features.csv` | Precomputed audio MFCC features keyed by song data. |
| `combined_similarity_matrix.csv` | Precomputed similarity matrix used by the command-line recommender. |
| `emotion_cnn_model.h5` | Final Keras emotion-classification model used by the API and live predictor. |
| `best_emotion_cnn_model.h5` | Best checkpoint produced during CNN training. |
| `emotion_cnn.onnx` | ONNX model artifact included for interoperability experiments. |

The FER2013 training images and the `archive/` dataset are intentionally not committed. See [Training the emotion model](#training-the-emotion-model) if you need them locally.

## Requirements

- Python 3.9 or newer
- Windows, macOS, or Linux
- A webcam for `/detect/emotion/process-video` and the live prediction scripts
- A local copy of the FER2013 directory only if retraining the model

The code uses Flask, Flask-CORS, OpenCV, NumPy, pandas, scikit-learn, TensorFlow/Keras, and the `fer` package for the experimental FER scripts.

## Installation

Create and activate a virtual environment from the repository root:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install flask flask-cors pandas numpy scikit-learn opencv-python tensorflow fer pillow
```

## Run the Flask API

From the repository root, start the service:

```bash
python app.py
```

The API starts at `http://localhost:5001`. On startup it loads `musicdata.csv`, computes the overall similarity matrix, and attempts to load lyric and audio feature matrices.

### Recommendation endpoints

Send a JSON object containing `liked_songs`. Song titles must match the `song_title` values in `musicdata.csv` exactly.

```bash
curl -X POST http://localhost:5001/recommend/overall ^
  -H "Content-Type: application/json" ^
  -d "{\"liked_songs\":[\"Chhudaina Timro Mayale\"]}"
```

The same request shape works with `/recommend/audio` and `/recommend/lyrics`:

```json
{
  "liked_songs": ["Chhudaina Timro Mayale"]
}
```

A successful response contains:

```json
{
  "recommendations": ["..."],
  "data": [
    {"title": "...", "score": 0.85}
  ],
  "based_on": ["Chhudaina Timro Mayale"]
}
```

If no valid liked title is supplied, the service returns an empty recommendation list. `POST /reload` reloads the CSV files and rebuilds the in-memory similarity matrices.

### Webcam emotion endpoint

The video endpoint is started with a POST request:

```bash
curl -X POST http://localhost:5001/detect/emotion/process-video
```

The server opens the machine's default webcam for up to 30 seconds. Press `Q` in the OpenCV window to stop early. A successful response includes the dominant emotion, confidence, number of analyzed frames, recording duration, and an emotion distribution. The process needs access to a graphical desktop and webcam, so it is not suitable for a headless server without additional changes.

## Command-Line Recommendation

`recommend_songs.py` uses the precomputed `combined_similarity_matrix.csv` without starting Flask:

```bash
python recommend_songs.py '["Chhudaina Timro Mayale"]'
```

It prints a JSON list containing up to five recommendations. The matrix file must contain song titles as both its index and columns.

## Training the Emotion Model

Place the FER2013 data in this layout before running the training script:

```text
fer2013/
  train/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
  test/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
```

Then run:

```bash
python train_emotion_model.py
```

The script trains for up to 50 epochs with augmentation, early stopping, learning-rate reduction, and checkpointing. It writes `best_emotion_cnn_model.h5` and `emotion_cnn_model.h5` to the repository root.

## Standalone Emotion Experiments

These scripts are useful for development and comparison, but they are not required to run the Flask API:

```bash
python predict.py
python live_emotion_predict.py
python newdet.py
python test1.py
```

`batch_detect.py` expects `DATASET_DIR` to be changed from its placeholder value before use. `detect_from_csv.py` expects `archive/ckextended.csv`, which is excluded from the repository.

## Data and Model Notes

- The application expects all CSV and model paths relative to the repository root.
- The audio feature loader parses the `mfccs_mean` column from a serialized Python-list representation.
- `combined_similarity_matrix.csv` is currently kept as a project artifact; the Flask API rebuilds its overall, lyric, and audio matrices at startup instead of loading the combined matrix.
- The checked-in H5 models are large binary artifacts. Git LFS or an external model registry may be preferable for future model versions.
- The dataset includes song metadata and feature artifacts; verify that you have permission to redistribute any music, lyrics, or derived data before publishing a production version.

## Current Limitations

- There is no frontend in this repository; clients must call the Flask endpoints directly.
- Webcam access and OpenCV display require a local graphical environment.
- The API uses exact song-title matching and returns an empty list for unknown titles.
- The training and experimental scripts use hard-coded relative paths and are intended for local experimentation.
- Dependency versions are not pinned yet, so TensorFlow/OpenCV installation may vary by operating system and Python version.

## License

No license file is currently included. Add an appropriate license before accepting external contributions or redistributing the project.
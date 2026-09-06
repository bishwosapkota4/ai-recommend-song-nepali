from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import subprocess
import logging
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
from collections import Counter

logging.basicConfig(level=logging.DEBUG)


app = Flask(__name__)
CORS(app)

# Load dataset
# Ensure you have a songs.csv file with columns: 'song_title', 'artist', 'genre', 'mood'
# DATA_FILE = 'musicdata.csv'
# df = None
# cosine_sim = None

# def load_data():
#     global df, cosine_sim
#     if os.path.exists(DATA_FILE):
#         try:
#             df = pd.read_csv(DATA_FILE)
#             # Create a combined features column for similarity
#             # Fill NaNs with empty string
#             df['features'] = df['genre'].fillna('') + ' ' + df['mood'].fillna('') + ' ' + df['artist'].fillna('')
            
#             tfidf = TfidfVectorizer(stop_words='english')
#             tfidf_matrix = tfidf.fit_transform(df['features'])
#             cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#             print("Data loaded and model trained.")
#         except Exception as e:
#             print(f"Error loading data: {e}")
#             df = None
#     else:
#         print(f"Warning: {DATA_FILE} not found. Recommendations will return empty.")

# load_data()

# def get_recommendations(song_titles, num_recommendations=5):
#     if df is None or cosine_sim is None:
#         return []
    
#     # Filter out songs that are not in our dataset
#     valid_songs = [song for song in song_titles if song in df['song_title'].values]
    
#     if not valid_songs:
#         return []

#     # Get indices of liked songs
#     indices = [df[df['song_title'] == song].index[0] for song in valid_songs]
    
#     # Calculate average similarity scores for all songs based on liked songs
#     # We sum the similarity scores for all liked songs
#     sim_scores = sum([cosine_sim[i] for i in indices])
    
#     # Get indices of sorted scores (descending)
#     sim_indices = sim_scores.argsort()[::-1]
    
#     recommended_songs = []
#     count = 0
#     for i in sim_indices:
#         title = df.iloc[i]['song_title']
#         if title not in song_titles: # Don't recommend songs the user already liked
#             recommended_songs.append(title)
#             count += 1
#             if count >= num_recommendations:
#                 break
                
#     return recommended_songs

# @app.route('/recommend', methods=['POST')
# def recommend():
#     data = request.get_json()
#     liked_songs = data.get('liked_songs', [])
    
#     if not liked_songs:
#         return jsonify({'recommendations': [], 'message': 'No liked songs provided'})

#     recommendations = get_recommendations(liked_songs)
    
#     return jsonify({
#         'recommendations': recommendations,
#         'based_on': liked_songs
#     })
# Load dataset
DATA_FILE = 'musicdata.csv'
LYRIC_FEATURES_FILE = 'lyric_features.csv'
AUDIO_FEATURES_FILE = 'audio_features.csv'
COMBINED_SIM_FILE = 'combined_similarity_matrix.csv'

df = None
cosine_sim_overall = None
cosine_sim_lyrics = None
cosine_sim_audio = None

# Update the `load_data` function to handle precomputed lyric features and parse audio features

def load_data():
    global df, cosine_sim_overall, cosine_sim_lyrics, cosine_sim_audio

    # Load Main Data
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            df['features'] = df['genre'].fillna('') + ' ' + df['mood'].fillna('') + ' ' + df['artist'].fillna('')

            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df['features'])
            cosine_sim_overall = cosine_similarity(tfidf_matrix, tfidf_matrix)
            print("Default overall model trained.")
            print("Cosine Similarity Overall Matrix Shape:", cosine_sim_overall.shape)
            print("Sample Values from Cosine Similarity Overall:", cosine_sim_overall[:5, :5])
        except Exception as e:
            print(f"Error loading main data: {e}")
            df = None
    else:
        print(f"Warning: {DATA_FILE} not found. Recommendations will return empty.")

    # Load Lyric Features
    if os.path.exists(LYRIC_FEATURES_FILE):
        try:
            lyric_df = pd.read_csv(LYRIC_FEATURES_FILE)
            feature_cols = [col for col in lyric_df.columns if col != 'song_title']
            if feature_cols:
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                lyric_features_scaled = scaler.fit_transform(lyric_df[feature_cols])
                cosine_sim_lyrics = cosine_similarity(lyric_features_scaled, lyric_features_scaled)
                print("Lyrics model trained using precomputed features.")
                print("Cosine Similarity Lyrics Matrix Shape:", cosine_sim_lyrics.shape)
                print("Sample Values from Cosine Similarity Lyrics:", cosine_sim_lyrics[:5, :5])
            else:
                print("No valid feature columns found in lyric features file.")
        except Exception as e:
            print(f"Error loading lyric features: {e}")

    # Load Audio Features
    if os.path.exists(AUDIO_FEATURES_FILE):
        try:
            audio_df = pd.read_csv(AUDIO_FEATURES_FILE)
            if 'mfccs_mean' in audio_df.columns:
                audio_df['mfccs_mean'] = audio_df['mfccs_mean'].apply(lambda x: eval(x) if isinstance(x, str) else x)
                mfcc_features = np.array(audio_df['mfccs_mean'].tolist())
                cosine_sim_audio = cosine_similarity(mfcc_features, mfcc_features)
                print("Audio model trained using parsed MFCC features.")
                print("Cosine Similarity Audio Matrix Shape:", cosine_sim_audio.shape)
                print("Sample Values from Cosine Similarity Audio:", cosine_sim_audio[:5, :5])
            else:
                print("'mfccs_mean' column not found in audio features file.")
        except Exception as e:
            print(f"Error loading audio features: {e}")

    # Load Combined Similarity Matrix if exists
    if os.path.exists(COMBINED_SIM_FILE):
        try:
            pass  # Placeholder for loading precomputed combined similarity matrix
        except Exception as e:
            print(f"Error loading combined similarity: {e}")

load_data()

def get_recommendations(song_titles, rec_type='overall', num_recommendations=5):
    if df is None:
        logging.debug("Main dataset (df) is not loaded.")
        return []

    # Select the appropriate similarity matrix based on the recommendation type
    if rec_type == 'overall':
        similarity_matrix = cosine_sim_overall
    elif rec_type == 'audio':
        similarity_matrix = cosine_sim_audio
    elif rec_type == 'lyrics':
        similarity_matrix = cosine_sim_lyrics
    else:
        logging.debug(f"Invalid recommendation type: {rec_type}")
        return []

    if similarity_matrix is None:
        logging.debug(f"Similarity matrix for {rec_type} is not loaded.")
        return []

    # Filter out songs that are not in our dataset
    valid_songs = [song for song in song_titles if song in df['song_title'].values]
    logging.debug(f"Valid songs from user input: {valid_songs}")

    if not valid_songs:
        logging.debug("No valid songs found in user input.")
        return []

    # Get indices of liked songs
    indices = [df[df['song_title'] == song].index[0] for song in valid_songs]
    logging.debug(f"Indices of liked songs: {indices}")

    # Calculate average similarity scores for all songs based on liked songs
    valid_indices = [i for i in indices if i < similarity_matrix.shape[0]]
    logging.debug(f"Valid indices within similarity matrix: {valid_indices}")

    if not valid_indices:
        logging.debug("No valid indices found within similarity matrix.")
        return []

    sim_scores = sum([similarity_matrix[i] for i in valid_indices])
    logging.debug(f"Similarity scores calculated: {sim_scores}")

    # Get indices of sorted scores (descending)
    sim_indices = sim_scores.argsort()[::-1]
    logging.debug(f"Sorted similarity indices: {sim_indices}")

    recommended_songs = []
    count = 0
    for i in sim_indices:
        if i >= len(df):
            continue

        title = df.iloc[i]['song_title']
        if title not in song_titles:  # Don't recommend songs the user already liked
            recommended_songs.append(title)
            count += 1
            if count >= num_recommendations:
                break

    logging.debug(f"Recommended songs: {recommended_songs}")
    return recommended_songs

@app.route('/reload', methods=['POST'])
def reload_data():
    load_data()
    return jsonify({'message': 'Data reloaded'})

# @app.route('/detect/emotion/process', methods=['POST'])
# def detect_emotion():
#     try:
#         # Call the live_emotion_predict.py script
#         process = subprocess.Popen(['python', 'live_emotion_predict.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         stdout, stderr = process.communicate()

#         if process.returncode != 0:
#             logging.error(stderr.decode('utf-8'))
#             return jsonify({"error": stderr.decode('utf-8')}), 500

#         return jsonify({"message": "Emotion detection script executed successfully."})

#     except Exception as e:
#         logging.error(str(e))
#         return jsonify({"error": str(e)}), 500

@app.route('/detect/emotion/process-video', methods=['POST'])
def detect_emotion_video():
    try:
        print("=== 🎬 Video Emotion Detection Started ===")
        
        # Check if model file exists
        model_path = 'emotion_cnn_model.h5'
        if not os.path.exists(model_path):
            return jsonify({
                "error": "Model file not found",
                "details": f"emotion_cnn_model.h5 not found at {os.path.abspath(model_path)}"
            }), 500

        # Load model only once
        if 'emotion_model' not in app.config:
            try:
                print("🔄 Loading emotion model...")
                app.config['emotion_model'] = load_model(model_path)
                print("✅ Emotion model loaded successfully")
            except Exception as model_error:
                return jsonify({
                    "error": "Failed to load emotion model",
                    "details": str(model_error)
                }), 500

        model = app.config['emotion_model']
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        print("✅ Model and labels ready")

        # Initialize webcam
        print("📹 Initializing webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not access webcam")
            return jsonify({"error": "Could not access webcam"}), 500

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Webcam configured successfully")

        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            cap.release()
            return jsonify({"error": "Face detection classifier not available"}), 500

        print("🎥 Starting video capture window...")
        print("💡 Instructions:")
        print("   - Record for 5-10 seconds with natural expressions")
        print("   - Ensure good lighting on your face")
        print("   - Window will close automatically after 10 seconds")
        print("   - Press 'Q' to close early")

        # Variables for emotion analysis
        emotion_predictions = []
        confidence_scores = []
        start_time = time.time()
        recording_duration = 30  # Changed from 10 to 30 seconds
        frame_count = 0
        frames_with_faces = 0

        while True:
            # Check if recording time is up
            current_time = time.time()
            elapsed = current_time - start_time
            if elapsed >= recording_duration:
                print("⏰ Recording time completed")
                break

            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from webcam")
                break

            frame_count += 1

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )

            current_emotion = "No face"
            current_confidence = 0

            if len(faces) > 0:
                frames_with_faces += 1
                # Use the largest face
                (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
                
                # Extract and preprocess face
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (48, 48))
                face_normalized = face_resized.reshape(1, 48, 48, 1).astype(np.float32) / 255.0

                # Predict emotion
                preds = model.predict(face_normalized, verbose=0)
                emotion_idx = np.argmax(preds)
                current_emotion = emotion_labels[emotion_idx]
                current_confidence = float(preds[0][emotion_idx])

                # Store prediction
                emotion_predictions.append(current_emotion)
                confidence_scores.append(current_confidence)

                # Draw rectangle and emotion text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                emotion_text = f"{current_emotion} ({current_confidence:.2f})"
                cv2.putText(frame, emotion_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # No face detected
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display timer and instructions
            remaining = recording_duration - elapsed
            cv2.putText(frame, f"Time: {remaining:.1f}s", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Frames with faces: {frames_with_faces}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'Q' to quit", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show the frame
            cv2.imshow('Video Emotion Detection - Recording...', frame)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 User pressed 'Q' - stopping recording")
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        print(f"✅ Video capture completed: {frame_count} total frames, {frames_with_faces} frames with faces")

        # Analyze results
        if frames_with_faces == 0:
            return jsonify({
                "error": "No faces detected in video",
                "message": "Please ensure your face is clearly visible during recording"
            }), 400

        # Determine dominant emotion
        if emotion_predictions:
            emotion_counter = Counter(emotion_predictions)
            dominant_emotion, dominant_count = emotion_counter.most_common(1)[0]
            
            # Calculate average confidence for dominant emotion
            dominant_confidences = [conf for emo, conf in zip(emotion_predictions, confidence_scores) if emo == dominant_emotion]
            avg_confidence = sum(dominant_confidences) / len(dominant_confidences) if dominant_confidences else 0
            
            emotion_distribution = dict(emotion_counter)
            
            print(f"🎭 Analysis Results:")
            print(f"   - Dominant emotion: {dominant_emotion}")
            print(f"   - Confidence: {avg_confidence:.3f}")
            print(f"   - Frames analyzed: {frames_with_faces}")
            print(f"   - Emotion distribution: {emotion_distribution}")

            return jsonify({
                "emotion": dominant_emotion,
                "confidence": round(avg_confidence, 3),
                "success": True,
                "message": f"Detected emotion: {dominant_emotion} from {frames_with_faces} video frames",
                "frames_analyzed": frames_with_faces,
                "recording_duration": round(elapsed, 2),
                "emotion_distribution": emotion_distribution,
                "analysis_method": "real_time_video_analysis"
            })
        else:
            return jsonify({
                "error": "No emotions detected",
                "message": "Please try again with clearer facial expressions"
            }), 400

    except Exception as e:
        print(f"💥 Video emotion detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Clean up resources
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        
        return jsonify({
            "error": "Video emotion detection failed",
            "details": str(e)
        }), 500

# @app.route('/api/recommend_songs', methods=['GET'])
# def recommend_songs_by_emotion():
#     try:
#         mood = request.args.get('mood', '').lower()
        
#         if not mood or df is None:
#             return jsonify([])
        
#         # Filter songs by mood
#         mood_songs = df[df['mood'].str.lower() == mood]
        
#         if mood_songs.empty:
#             # If no exact match, try partial match
#             mood_songs = df[df['mood'].str.lower().str.contains(mood, na=False)]
        
#         # Convert to list of dictionaries
#         recommendations = mood_songs.head(10).to_dict('records')
        
#         return jsonify(recommendations)
        
#     except Exception as e:
#         logging.error(f"Song recommendation error: {str(e)}")
#         return jsonify([])

# Add endpoints for different recommendation types
@app.route('/recommend/overall', methods=['POST'])
def recommend_overall():
    data = request.get_json()
    liked_songs = data.get('liked_songs', [])

    if not liked_songs:
        return jsonify({'recommendations': [], 'message': 'No liked songs provided'})

    recommendations_data = get_recommendations_with_scores(liked_songs, rec_type='overall')
    recommendations = [item['title'] for item in recommendations_data]

    return jsonify({
        'recommendations': recommendations,
        'data': recommendations_data,
        'based_on': liked_songs
    })

@app.route('/recommend/audio', methods=['POST'])
def recommend_audio():
    data = request.get_json()
    liked_songs = data.get('liked_songs', [])

    if not liked_songs:
        return jsonify({'recommendations': [], 'message': 'No liked songs provided'})

    recommendations_data = get_recommendations_with_scores(liked_songs, rec_type='audio')
    recommendations = [item['title'] for item in recommendations_data]

    return jsonify({
        'recommendations': recommendations,
        'data': recommendations_data,
        'based_on': liked_songs
    })

@app.route('/recommend/lyrics', methods=['POST'])
def recommend_lyrics():
    data = request.get_json()
    liked_songs = data.get('liked_songs', [])

    if not liked_songs:
        return jsonify({'recommendations': [], 'message': 'No liked songs provided'})

    recommendations_data = get_recommendations_with_scores(liked_songs, rec_type='lyrics')
    recommendations = [item['title'] for item in recommendations_data]

    return jsonify({
        'recommendations': recommendations,
        'data': recommendations_data,
        'based_on': liked_songs
    })

def get_recommendations_with_scores(song_titles, rec_type='overall', num_recommendations=5):
    if df is None:
        return []

    current_sim = cosine_sim_overall
    if rec_type == 'lyrics' and cosine_sim_lyrics is not None:
        current_sim = cosine_sim_lyrics
    elif rec_type == 'audio' and cosine_sim_audio is not None:
        current_sim = cosine_sim_audio

    if current_sim is None:
        current_sim = cosine_sim_overall

    if current_sim is None:
        return []

    valid_songs = [song for song in song_titles if song in df['song_title'].values]

    if not valid_songs:
        return []

    indices = [df[df['song_title'] == song].index[0] for song in valid_songs]
    valid_indices = [i for i in indices if i < current_sim.shape[0]]

    if not valid_indices:
        return []

    sim_scores = sum([current_sim[i] for i in valid_indices])
    sim_indices = sim_scores.argsort()[::-1]

    recommended_data = []
    count = 0
    for i in sim_indices:
        if i >= len(df):
            continue

        title = df.iloc[i]['song_title']
        if title not in song_titles:
            recommended_data.append({
                'title': title,
                'score': float(sim_scores[i])
            })
            count += 1
            if count >= num_recommendations:
                break

    return recommended_data

# @app.route('/shutdown', methods=['POST'])
# def shutdown():
#     # Optional: Add a security token for authentication
#     token = request.headers.get('Authorization')
#     if token != "your-secure-token":
#         return jsonify({'error': 'Unauthorized access'}), 403

#     func = request.environ.get('werkzeug.server.shutdown')
#     if func is None:
#         # If running with a different WSGI server or newer Flask/Werkzeug
#         os._exit(0)
#     func()
#     return jsonify({'message': 'Server shutting down...'})

if __name__ == '__main__':
    # Run on port 5001 to avoid conflict with your existing app on 5080
    app.run(debug=True, port=5001)

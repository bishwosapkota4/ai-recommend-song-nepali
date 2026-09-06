from fer.fer import FER
import cv2
import time

class RealTimeFER:
    def __init__(self, mtcnn=True, processing_interval=2):
        self.detector = FER(mtcnn=mtcnn)
        self.processing_interval = processing_interval  # Process every Nth frame
        self.frame_count = 0
        self.last_results = []
        
    def process_frame(self, frame):
        self.frame_count += 1
        
        # Only process every Nth frame to improve performance
        if self.frame_count % self.processing_interval == 0:
            try:
                self.last_results = self.detector.detect_emotions(frame)
            except Exception as e:
                print(f"Processing error: {e}")
                self.last_results = []
        
        return self.last_results
    
    def draw_results(self, frame, results):
        for result in results:
            x, y, w, h = result["box"]
            emotions = result["emotions"]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display top 3 emotions
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            
            y_offset = y - 30
            for emotion, score in sorted_emotions:
                if score > 0.1:
                    text = f"{emotion}: {score:.2f}"
                    cv2.putText(frame, text, (x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset -= 25
        
        return frame

# Usage
fer_processor = RealTimeFER(mtcnn=False, processing_interval=3)  # Process every 3rd frame, use Haar Cascade for speed

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    
    # Process frame
    results = fer_processor.process_frame(frame)
    
    # Draw results
    frame_with_results = fer_processor.draw_results(frame, results)
    
    cv2.imshow('Optimized FER + MTCNN', frame_with_results)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
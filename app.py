from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import math
import threading
import time
import os
import pyaudio
import wave
import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from face_detector import get_face_detector, find_faces, draw_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks
import tensorflow as tf

app = Flask(__name__)

# Global variables
global cap, audio_thread, is_recording
cap = None
audio_thread = None
is_recording = False
violations = {
    "face_missing": 0,
    "multiple_faces": 0,
    "looking_away": 0,
    "speaking": 0,
    "mouth_open": 0
}
violation_thresholds = {
    "face_missing": 5,
    "multiple_faces": 3,
    "looking_away": 5,
    "speaking": 3,
    "mouth_open": 3
}

# Initialize the models
face_model = get_face_detector()
landmark_model = get_landmark_model()

# Initialize mouth parameters
outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0]*5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0]*3

# Head pose parameters
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Eye tracking parameters
left_eye = [36, 37, 38, 39, 40, 41]
right_eye = [42, 43, 44, 45, 46, 47]
kernel = np.ones((9, 9), np.uint8)

# Audio parameters
chunk = 1024
sample_format = pyaudio.paInt16
channels = 2
fs = 44100
seconds = 5
p = pyaudio.PyAudio()

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def get_frame():
    """Return the current camera frame."""
    global cap, violations
    
    if cap is None:
        return None
    
    ret, frame = cap.read()
    if not ret:
        return None
    
    # Get frame dimensions for camera matrix
    size = frame.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    
    # Find faces
    faces = find_faces(frame, face_model)
    
    # Check for face violations
    if len(faces) == 0:
        violations["face_missing"] += 1
        cv2.putText(frame, "NO FACE DETECTED", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif len(faces) > 1:
        violations["multiple_faces"] += 1
        cv2.putText(frame, "MULTIPLE FACES DETECTED", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        violations["face_missing"] = max(0, violations["face_missing"] - 1)
        violations["multiple_faces"] = max(0, violations["multiple_faces"] - 1)
    
    # Process each detected face
    for face in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)
        
        # Detect landmarks
        shape = detect_marks(frame, landmark_model, face)
        
        # Draw landmarks
        draw_marks(frame, shape)
        
        # Head pose estimation
        image_points = np.array([
            shape[30],     # Nose tip
            shape[8],      # Chin
            shape[36],     # Left eye left corner
            shape[45],     # Right eye right corner
            shape[48],     # Left Mouth corner
            shape[54]      # Right mouth corner
        ], dtype="double")
        
        dist_coeffs = np.zeros((4,1))
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP
        )
        
        # Project a 3D point onto the image plane
        (nose_end_point2D, jacobian) = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs
        )
        
        # Draw head pose line
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.line(frame, p1, p2, (0, 255, 255), 2)
        
        # Calculate angles for head pose
        try:
            m = (p2[1] - p1[1])/(p2[0] - p1[0])
            ang1 = int(math.degrees(math.atan(m)))
        except:
            ang1 = 90
            
        # Get side points for horizontal angle
        rear_size = 1
        rear_depth = 0
        front_size = frame.shape[1]
        front_depth = front_size*2
        val = [rear_size, rear_depth, front_size, front_depth]
        
        point_2d = get_2d_points(frame, rotation_vector, translation_vector, camera_matrix, val)
        y = (point_2d[5] + point_2d[8])//2
        x = point_2d[2]
        
        cv2.line(frame, tuple(x), tuple(y), (255, 255, 0), 2)
        
        try:
            m = (y[1] - x[1])/(y[0] - x[0])
            ang2 = int(math.degrees(math.atan(-1/m)))
        except:
            ang2 = 90
            
        # Check head orientation
        looking_away = False
        if ang1 >= 48 or ang1 <= -48 or ang2 >= 48 or ang2 <= -48:
            looking_away = True
            violations["looking_away"] += 1
            cv2.putText(frame, "LOOKING AWAY", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            violations["looking_away"] = max(0, violations["looking_away"] - 1)
            
        # Eye tracking
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask, end_points_left = eye_on_mask(mask, left_eye, shape)
        mask, end_points_right = eye_on_mask(mask, right_eye, shape)
        mask = cv2.dilate(mask, kernel, 5)
        
        eyes = cv2.bitwise_and(frame, frame, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = int((shape[42][0] + shape[39][0]) // 2)
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(eyes_gray, 75, 255, cv2.THRESH_BINARY)
        thresh = process_thresh(thresh)
        
        eyeball_pos_left = contouring(thresh[:, 0:mid], mid, frame, end_points_left)
        eyeball_pos_right = contouring(thresh[:, mid:], mid, frame, end_points_right, True)
        
        # Mouth detection
        mouth_open = False
        cnt_outer = 0
        cnt_inner = 0
        
        # If mouth distances have been calibrated
        if sum(d_outer) > 0 and sum(d_inner) > 0:
            for i, (p1, p2) in enumerate(outer_points):
                if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                    cnt_outer += 1 
            for i, (p1, p2) in enumerate(inner_points):
                if d_inner[i] + 2 < shape[p2][1] - shape[p1][1]:
                    cnt_inner += 1
                    
            if cnt_outer > 3 and cnt_inner > 2:
                mouth_open = True
                violations["mouth_open"] += 1
                cv2.putText(frame, "MOUTH OPEN", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                violations["mouth_open"] = max(0, violations["mouth_open"] - 1)
    
    # Add violation status on the frame
    cv2.putText(frame, f"Violations: {sum(1 for k, v in violations.items() if v >= violation_thresholds[k])}", 
                (30, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    import logging

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and a stream handler
file_handler = logging.FileHandler('proctoring.log')
stream_handler = logging.StreamHandler()

# Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Replace print statements with logger statements
def get_frame():
    """Return the current camera frame."""
    global cap, violations
    
    if cap is None:
        logger.info("Camera not initialized")
        return None
    
    ret, frame = cap.read()
    if not ret:
        logger.info("Failed to read camera frame")
        return None
    
    # ... rest of the function ...

def read_audio(stream, filename):
    """Record audio for a specified duration and save it to a file."""
    frames = []
    
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    
    logger.info(f"Audio recorded to {filename}")

def convert_audio_to_text(i):
    """Convert audio file to text using Google Speech Recognition."""
    global violations
    
    if i >= 0:
        sound = f'static/audio/record{i}.wav'
        r = sr.Recognizer()
        
        with sr.AudioFile(sound) as source:
            r.adjust_for_ambient_noise(source)
            logger.info(f"Converting Audio To Text: {sound}")
            audio = r.listen(source)
        
        try:
            value = r.recognize_google(audio)
            os.remove(sound)
            
            if str is bytes:
                result = u"{}".format(value).encode("utf-8")
            else:
                result = "{}".format(value)
                
            logger.info(f"Speech detected: {result}")
            
            # If speech is detected, increment violation
            if result.strip():
                violations["speaking"] += 1
                logger.info("Speaking violation increased")
            else:
                violations["speaking"] = max(0, violations["speaking"] - 1)
                
            with open("static/text/speech_log.txt", "a") as f:
                f.write(result)
                f.write(" ")
                
            return result
                
        except sr.UnknownValueError:
            logger.info("No speech detected")
            violations["speaking"] = max(0, violations["speaking"] - 1)
            return ""
        except sr.RequestError as e:
            logger.info(f"Error with speech recognition service: {e}")
            return ""
        except Exception as e:
            logger.info(f"Error processing audio: {e}")
            return ""

def audio_monitoring():
    """Continuously monitor audio in a separate thread."""
    global is_recording
    
    logger.info("Starting audio monitoring...")
    i = 0
    
    while is_recording:
        try:
            stream = p.open(format=sample_format, channels=channels, rate=fs,
                      frames_per_buffer=chunk, input=True)
            
            filename = f'static/audio/record{i}.wav'
            logger.info(f"Recording audio to {filename}")
            read_audio(stream, filename)
            
            # Process the recorded audio
            speech_text = convert_audio_to_text(i)
            
            i += 1
            time.sleep(1)  # Small delay between recordings
            
        except Exception as e:
            logger.info(f"Error in audio monitoring: {e}")
            time.sleep(1)
            
    logger.info("Audio monitoring stopped")

# ... rest of the code ...
    # Encode the frame for streaming
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

def read_audio(stream, filename):
    """Record audio for a specified duration and save it to a file."""
    frames = []
    
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()

def convert_audio_to_text(i):
    """Convert audio file to text using Google Speech Recognition."""
    global violations
    
    if i >= 0:
        sound = f'static/audio/record{i}.wav'
        r = sr.Recognizer()
        
        with sr.AudioFile(sound) as source:
            r.adjust_for_ambient_noise(source)
            print(f"Converting Audio To Text: {sound}")
            audio = r.listen(source)
        
        try:
            value = r.recognize_google(audio)
            os.remove(sound)
            
            if str is bytes:
                result = u"{}".format(value).encode("utf-8")
            else:
                result = "{}".format(value)
                
            print(f"Speech detected: {result}")
            
            # If speech is detected, increment violation
            if result.strip():
                violations["speaking"] += 1
                print("Speaking violation increased")
            else:
                violations["speaking"] = max(0, violations["speaking"] - 1)
                
            with open("static/text/speech_log.txt", "a") as f:
                f.write(result)
                f.write(" ")
                
            return result
                
        except sr.UnknownValueError:
            print("No speech detected")
            violations["speaking"] = max(0, violations["speaking"] - 1)
            return ""
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")
            return ""
        except Exception as e:
            print(f"Error processing audio: {e}")
            return ""

def audio_monitoring():
    """Continuously monitor audio in a separate thread."""
    global is_recording
    
    print("Starting audio monitoring...")
    i = 0
    
    while is_recording:
        try:
            stream = p.open(format=sample_format, channels=channels, rate=fs,
                      frames_per_buffer=chunk, input=True)
            
            filename = f'static/audio/record{i}.wav'
            print(f"Recording audio to {filename}")
            read_audio(stream, filename)
            
            # Process the recorded audio
            speech_text = convert_audio_to_text(i)
            
            i += 1
            time.sleep(1)  # Small delay between recordings
            
        except Exception as e:
            print(f"Error in audio monitoring: {e}")
            time.sleep(1)
            
    print("Audio monitoring stopped")

# Helper functions from the original code
def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    """Return the 3D points present as 2D for making annotation box"""
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    
    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float64).reshape(-1, 3)
    
    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def eye_on_mask(mask, side, shape):
    """Create mask from eye landmarks"""
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1]+points[2][1])//2
    r = points[3][0]
    b = (points[4][1]+points[5][1])//2
    return mask, [l, t, r, b]

def find_eyeball_position(end_points, cx, cy):
    """Find and classify the eyeball position"""
    x_ratio = (end_points[0] - cx)/(cx - end_points[2])
    y_ratio = (cy - end_points[1])/(end_points[3] - cy)
    if x_ratio > 3:
        return 1  # Looking left
    elif x_ratio < 0.33:
        return 2  # Looking right
    elif y_ratio < 0.33:
        return 3  # Looking up
    else:
        return 0  # Looking center

def contouring(thresh, mid, img, end_points, right=False):
    """Find contours in the thresholded eye image"""
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        return 0

def process_thresh(thresh):
    """Process thresholded eye image for better contour detection"""
    thresh = cv2.erode(thresh, None, iterations=2) 
    thresh = cv2.dilate(thresh, None, iterations=4) 
    thresh = cv2.medianBlur(thresh, 3) 
    thresh = cv2.bitwise_not(thresh)
    return thresh

def calibrate_mouth():
    """Calibrate mouth parameters for the current user"""
    global cap, d_outer, d_inner
    
    if cap is None:
        return {"success": False, "message": "Camera not initialized"}
    
    # Reset mouth distances
    d_outer = [0]*5
    d_inner = [0]*3
    
    # Collect 30 frames for calibration
    for i in range(30):
        ret, img = cap.read()
        if not ret:
            return {"success": False, "message": "Failed to read camera frame"}
        
        faces = find_faces(img, face_model)
        if not faces:
            continue
            
        face = faces[0]  # Use the first detected face
        shape = detect_marks(img, landmark_model, face)
        
        for i, (p1, p2) in enumerate(outer_points):
            d_outer[i] += shape[p2][1] - shape[p1][1]
            
        for i, (p1, p2) in enumerate(inner_points):
            d_inner[i] += shape[p2][1] - shape[p1][1]
            
        time.sleep(0.1)  # Small delay between frames
    
    # Average the distances
    d_outer[:] = [x / 30 for x in d_outer]
    d_inner[:] = [x / 30 for x in d_inner]
    
    return {"success": True, "message": "Mouth calibration completed"}

def reset_violations():
    """Reset all violation counters"""
    global violations
    
    for key in violations:
        violations[key] = 0
    
    return {"success": True, "message": "Violations reset"}

@app.route('/')
def index():
    """Render the index page"""
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_proctoring():
    """Start the proctoring system"""
    global cap, audio_thread, is_recording
    
    try:
        # Initialize camera
        if cap is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return jsonify({"success": False, "message": "Failed to open camera"})
        
        # Start audio monitoring thread
        if audio_thread is None or not audio_thread.is_alive():
            # Ensure directories exist
            os.makedirs('static/audio', exist_ok=True)
            os.makedirs('static/text', exist_ok=True)
            
            is_recording = True
            audio_thread = threading.Thread(target=audio_monitoring)
            audio_thread.daemon = True
            audio_thread.start()
        
        # Reset violations
        reset_violations()
        
        return jsonify({"success": True, "message": "Proctoring started"})
    
    except Exception as e:
        return jsonify({"success": False, "message": f"Error starting proctoring: {str(e)}"})

@app.route('/stop', methods=['POST'])
def stop_proctoring():
    """Stop the proctoring system"""
    global cap, audio_thread, is_recording
    
    try:
        # Stop audio recording
        is_recording = False
        if audio_thread and audio_thread.is_alive():
            audio_thread.join(timeout=2)
            audio_thread = None
        
        # Release camera
        if cap:
            cap.release()
            cap = None
        
        return jsonify({"success": True, "message": "Proctoring stopped"})
    
    except Exception as e:
        return jsonify({"success": False, "message": f"Error stopping proctoring: {str(e)}"})

@app.route('/calibrate_mouth', methods=['POST'])
def calibrate_mouth_route():
    """Route to calibrate mouth parameters"""
    result = calibrate_mouth()
    return jsonify(result)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(
        (b'--frame\r\n'
         b'Content-Type: image/jpeg\r\n\r\n' + get_frame() + b'\r\n'),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/get_violations')
def get_violations():
    """Return current violation counts"""
    global violations, violation_thresholds
    
    # Count serious violations (those over threshold)
    serious_violations = sum(1 for k, v in violations.items() if v >= violation_thresholds[k])
    
    return jsonify({
        "violations": violations,
        "thresholds": violation_thresholds,
        "serious_count": serious_violations
    })

if __name__ == '__main__':
    app.run(debug=True)
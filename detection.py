import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np


model = load_model("emotion_model.h5")
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "happy", 4: "neutral", 5: "sad",6:"surpeise"}

# Initialize emotion counters
emo_counters = {"angry": 0, "disgust": 0, "fear": 0,"happy": 0, "neutral": 0, "sad": 0, "surpise": 0}
frame_per = 0

def check_nervous(emotion, emo_counters):
    if emotion == emotion_dict[0]:
        emo_counters["angry"] += 1
    elif emotion == emotion_dict[1]:
        emo_counters["disgust"] += 1
    elif emotion == emotion_dict[2]:
        emo_counters["fear"] += 1
    elif emotion == emotion_dict[3]:
        emo_counters["happy"] += 1    
    elif emotion == emotion_dict[4]:
        emo_counters["neutral"] += 1
    elif emotion == emotion_dict[5]:
        emo_counters["sad"] += 1    
    else:
        emo_counters["surpise"] += 1
    return emo_counters

def predict_emotion(face_image):
    if len(face_image.shape) == 2:
        pass
    else:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = cv2.resize(face_image, (48, 48))
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
    face_image = face_image / 255.0
    
    emotion = np.argmax(model.predict(face_image))
    return emotion_dict[emotion]

def face_detect(frame, frame_per, emo_counters):
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        image = frame
        image.flags.writeable = False
        results = face_detection.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
                ih, iw, _ = image.shape
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                
                face = image[y:y+h, x:x+w]
                if face.size > 0:
                    emotion = predict_emotion(face)
                    emo_counters = check_nervous(emotion, emo_counters)
                    frame_per += 0.5
                    cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    emotion = "No Face"
                    cv2.putText(image, emotion, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return image, frame_per


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    

    processed_frame, frame_per = face_detect(frame, frame_per, emo_counters)
    

    cv2.imshow('Emotion Detection', processed_frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

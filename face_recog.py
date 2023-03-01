import cv2
import face_recognition
import os
import glob
import pyttsx3

# Directory containing custom dataset
dataset_dir = "custom_dataset"

# Load images for each person in dataset
known_face_encodings = []
known_face_names = []
for person_dir in os.listdir(dataset_dir):
    person_name = os.path.basename(person_dir)
    for image_path in glob.glob(os.path.join(dataset_dir, person_dir, "*.jpg")):
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            known_face_encodings.append(face_encodings[0])
            known_face_names.append(person_name)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Set voice properties
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id) # Change index to select a different voice

# Capture frames from webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1] # Convert BGR to RGB
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Recognize faces
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            name = "Unknown"

        print("Found face:", name)

        # Speak name of recognized person
        engine.say(f"I see {name}")
        engine.runAndWait()

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

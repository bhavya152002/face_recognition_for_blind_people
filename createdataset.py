import cv2
import os

# Directory to store captured images
dataset_dir = "upworkWrok2(face)/dataset"

# Create dataset directory if it doesn't exist
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Initialize variables
person_name = ""
image_count = 0

while True:
    # Capture frame from webcam
    ret, frame = video_capture.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Draw bounding boxes around detected faces and capture images
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # If person name is not set, prompt user to enter name
        if not person_name:
            cv2.putText(frame, "Enter name and press 'Enter'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if cv2.waitKey(1) & 0xFF == ord('\n'):
                person_name = input("Enter name: ")
                print("Capturing images for", person_name)

        # If person name is set, capture image and save to dataset
        else:
            image_path = os.path.join(dataset_dir, person_name, f"{person_name}_{image_count}.jpg")
            cv2.imwrite(image_path, gray_frame[y:y+h, x:x+w])
            image_count += 1

    # Display frame
    cv2.imshow('Video', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
video_capture.release()
cv2.destroyAllWindows()

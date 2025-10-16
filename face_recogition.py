# Emotion Detection using Webcam
# Works in PyCharm
# Detects: happy, sad, angry, surprise, fear, disgust, neutral

import cv2
from deepface import DeepFace

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    try:
        # Analyze the frame for emotion
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Get the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Display the emotion text on the frame
        cv2.putText(frame, f"Emotion: {emotion}", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    except Exception as e:
        # If no face is detected
        cv2.putText(frame, "No face detected", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Show the video frame
    cv2.imshow("Emotion Detection", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close window
cap.release()
cv2.destroyAllWindows()

import cv2
from keras.models import model_from_json
import numpy as np
import os

# Load the pre-trained model
json_file = open("c:/Users/hp/Programming VSC/PROJECTS/Emotion Detection/emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("c:/Users/hp/Programming VSC/PROJECTS/Emotion Detection/emotiondetector.h5")

# Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels
labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Path to emoji images corresponding to each emotion (just filenames)
emoji_images = {
    'Angry': 'angry.png',
    'Disgust': 'disgust.png',
    'Fear': 'fear.png',
    'Happy': 'happy.png',
    'Neutral': 'neutral.png',
    'Sad': 'sad.png',
    'Surprise': 'surprise.png'
}

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Function to overlay emoji
def overlay_emoji(frame, emoji_name, position, size=(50, 50)):
    # Construct absolute path to the emoji file
    emoji_dir = r"C:\Users\hp\Programming VSC\PROJECTS\Emotion Detection\emoji_images"
    emoji_path = os.path.join(emoji_dir, emoji_name)  # Only filename from emoji_images
    
    if not os.path.exists(emoji_path):
        print(f"File not found: {emoji_path}")
        return
    
    # Load the emoji image
    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    if emoji is None:
        print(f"Error loading image: {emoji_path}")
        return
    emoji_resized = cv2.resize(emoji, size)

    # Get the region of interest where the emoji will be placed
    x, y = position
    h, w = emoji_resized.shape[:2]
    # Check if the ROI is valid before overlaying
    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        print(f"Overlay position {position} is out of bounds of the frame")
        return
    
    roi = frame[y:y+h, x:x+w]

    # Convert emoji to grayscale if it has an alpha channel (for transparency)
    if emoji_resized.shape[2] == 4:
        alpha = emoji_resized[:, :, 3] / 255.0
        for c in range(0, 3):
            roi[:, :, c] = (alpha * emoji_resized[:, :, c] + (1 - alpha) * roi[:, :, c])
    else:
        # If no alpha channel, simply overlay the emoji
        frame[y:y+h, x:x+w] = emoji_resized

    # Place the updated ROI back into the frame
    frame[y:y+h, x:x+w] = roi  # Ensure the frame is updated correctly

# Initialize webcam
webcam = cv2.VideoCapture(0)
cv2.namedWindow("Real-Time Emotion Detector", cv2.WINDOW_NORMAL)  # Create a resizable window

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Detect faces and predict emotions
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img_resized = cv2.resize(face_img, (48, 48))
        img = extract_features(face_img_resized)

        # Predict emotion and get confidence score
        predictions = model.predict(img)
        prediction_label = labels[predictions.argmax()]
        confidence = predictions.max() * 100  # Confidence in percentage

        # Get the corresponding emoji file name
        emoji_filename = emoji_images[prediction_label]

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # # Add emotion label and confidence percentage
        # label_text = f"{prediction_label} ({confidence:.2f}%)"
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # label_position = (x, y - 10)

        # # Display emotion label with text
        # cv2.putText(frame, label_text, label_position, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Add emotion label, emoji, and confidence percentage
        label_text = prediction_label
        confidence_text = f"({confidence:.2f}%)"

        # Define font and position for label
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_position = (x, y - 10)

        # Display emotion label with emoji
        cv2.putText(frame, label_text, label_position, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display smaller confidence percentage below the label
        confidence_position = (x, y + 20)
        cv2.putText(frame, confidence_text, confidence_position, font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        # Overlay the emoji on the frame
        emoji_size = (50, 50)  # Adjust size of the emoji
        overlay_emoji(frame, emoji_filename, (x+w+10, y), size=emoji_size)

    # Display the output
    cv2.imshow("Real-Time Emotion Detector", frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()

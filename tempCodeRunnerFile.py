json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("c:/Users/hp/Programming VSC/PROJECTS/Emotion Detection/emotiondetector.h5")

# Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels
labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

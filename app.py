import os
from flask import Flask, render_template, request
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
MESSAGE = "WELCOME  " \
          " Instruction: Choose an option to unlock the door or add a new user."
# Defining Flask App
app = Flask(__name__)
# Initializing VideoCapture object to access Webcam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Attempt to open the camera
cap = cv2.VideoCapture(0)
# If the camera fails to open, print an error message
if not cap.isOpened():
    print("Error: Could not open camera.")
    # You may want to add further error handling or exit the program here
# If these directories don't exist, create them
if not os.path.isdir('static'):
    os.makedirs('static/faces')
# Global variable to track door status (locked or unlocked)
door_locked = True
# Get the number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))
# Extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points
# Identify face using ML model
# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    label = model.predict(facearray)
    # Assuming the label is in the format 'username_userid', split it to get the name
    full_name = label[0].split('_')[0]
    return full_name
# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')
    def send_email(subject, body):
        msg = MIMEMultipart()
        msg['From'] = "venkateshdumpa90@gmail.com"
        msg['To'] = "venkateshdumpa90@gmail.com"
        msg['Subject'] = 'hello'
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login("venkateshdumpa90@gmail.com","venky@2004")
            smtp.send_message(msg)
# Our main page
@app.route('/')
def home():
    global MESSAGE
    return render_template('index.html', mess=MESSAGE)
@app.route('/unlock_door', methods=['GET'])
def unlock_door():
    global door_locked, MESSAGE
    # If the camera was not opened before, attempt to open it now
    if not cap.isOpened():
        cap.open(0)
    identified_person = ""
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from the camera.")
            break
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            cv2.putText(frame, f'{identified_person}',(x + 6, y - 6),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)
        # Display the resulting frame
        cv2.imshow('Unlocking Door, press "q" to exit', frame)
        # If no face is recognized, show an error message
        if not identified_person:
            MESSAGE = 'Error: No face detected. Please try again.'
            break
        # If a face is recognized, unlock the door
        if identified_person:
            MESSAGE = f'Door unlocked for {identified_person}. Choose an option.'
            break
        # Wait for the user to press 'q' to quit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    # Release the camera when done
    cap.release()
    cv2.destroyAllWindows()
    return render_template('index.html', mess=MESSAGE)
# This function will run when we click on Add New User option
@app.route('/add_new_user', methods=['POST'])
def add_new_user():
    new_user_name = request.form['new_user_name']
    user_image_folder = f'static/faces/{new_user_name}'
    if not os.path.isdir(user_image_folder):
        os.makedirs(user_image_folder)
    cap = cv2.VideoCapture(0)  # Use the webcam for face capture
    i = 0
    while i < 400:  # Capture 50 frames for training
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y),(x+w, y+h),(255, 0, 20), 2)
            if i % 20 == 0:
                image_name = f'{new_user_name}_{i}.jpg'
                cv2.imwrite(os.path.join(user_image_folder, image_name), frame[y:y+h, x:x+w])
            i += 1
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:  # Press 'Esc' to stop capturing
            break
    # Release the camera when done
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    return render_template('index.html', mess=f'User {new_user_name} added successfully. Choose an option.')
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=1000)

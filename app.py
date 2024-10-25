from __future__ import division, print_function
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from flask import Flask, request, render_template
import statistics as st
import time
import argparse
import eel
import random
import glob
from PIL import Image, ImageTk
import tkinter as tk

# Custom class for DepthwiseConv2D to handle 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove 'groups' if it exists
        super().__init__(*args, **kwargs)

# Define custom object mapping for model loading
custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}

# Load the model with custom objects once
model = load_model('final_model.h5', custom_objects=custom_objects)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index1.html")

@app.route('/camera', methods=['GET', 'POST'])
def camera():
    GR_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    output = []
    cap = cv2.VideoCapture(0)
    
    i = 0
    while i <= 30:
        ret, img = cap.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(img, 1.05, 5)
        
        for x, y, w, h in faces:
            face_img = img[y:y+h, x:x+w]
            resized = cv2.resize(face_img, (224, 224))
            reshaped = resized.reshape(1, 224, 224, 3) / 255
            predictions = model.predict(reshaped)

            max_index = np.argmax(predictions[0])
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
            predicted_emotion = emotions[max_index]
            output.append(predicted_emotion)
            
            cv2.rectangle(img, (x, y), (x+w, y+h), GR_dict[1], 2)
            cv2.rectangle(img, (x, y-40), (x+w, y), GR_dict[1], -1)
            cv2.putText(img, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            break
        
        i += 1

    cap.release()
    cv2.destroyAllWindows()

    if output:
        final_output1 = st.mode(output)
    else:
        final_output1 = 'unknown'
    
    return render_template("buttons.html", final_output=final_output1)

@app.route('/templates/buttons', methods=['GET', 'POST'])
def buttons():
    return render_template("buttons.html")

@app.route('/movies/<emotion>', methods=['GET', 'POST'])
def movies(emotion):
    valid_emotions = {'surprise', 'angry', 'sad', 'disgust', 'happy', 'fear', 'neutral'}
    if emotion in valid_emotions:
        return render_template(f"movies{emotion.capitalize()}.html")
    else:
        return "Invalid emotion", 404

@app.route('/songs/<emotion>', methods=['GET', 'POST'])
def songs(emotion):
    valid_emotions = {'surprise', 'angry', 'sad', 'disgust', 'happy', 'fear', 'neutral'}
    if emotion in valid_emotions:
        return render_template(f"songs{emotion.capitalize()}.html")
    else:
        return "Invalid emotion", 404

@app.route('/templates/join_page.html', methods=['GET', 'POST'])
def join():
    return render_template("join_page.html")

# Additional code to handle face recognition and model updates
fishface = cv2.face.FisherFaceRecognizer_create()
facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
facedict = {}
eel.init('WD_INNOVATIVE')

def crop(clahe_image, face):
    for (x, y, w, h) in face:
        faceslice = clahe_image[y:y+h, x:x+w]
        faceslice = cv2.resize(faceslice, (350, 350))
        facedict["face%s" %(len(facedict)+1)] = faceslice
    return faceslice

def grab_face():
    ret, frame = video_capture.read()
    cv2.imwrite('test.jpg', frame)
    gray = cv2.imread('test.jpg', 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    return clahe_image

def detect_face():
    clahe_image = grab_face()
    face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face) >= 1:
        faceslice = crop(clahe_image, face)
    else:
        print("No/Multiple faces detected!!, passing over the frame")

def save_face(emotion):
    print("\n\nLook "+emotion+" until the timer expires and keep the same emotion for some time.")
    
    for i in range(0, 5):
        print(5-i)
        time.sleep(1)
    
    while len(facedict.keys()) < 16:
        detect_face()

    for i in facedict.keys():
        path, dirs, files = next(os.walk("dataset/%s" %emotion))
        file_count = len(files) + 1
        cv2.imwrite("dataset/%s/%s.jpg" %(emotion, (file_count)), facedict[i])
    facedict.clear()

def update_model(emotions):
    print("Update mode for model is ready")
    checkForFolders(emotions)
    
    for i in range(0, len(emotions)):
        save_face(emotions[i])
    print("Collected the images, looking nice! Now updating the model...")
    Update_Model.update(emotions)
    print("Model train successful!!")

def checkForFolders(emotions):
    for emotion in emotions:
        if os.path.exists("dataset/%s" %emotion):
            pass
        else:
            os.makedirs("dataset/%s" %emotion)

def identify_emotions():
    prediction = []
    confidence = []

    for i in facedict.keys():
        pred, conf = fishface.predict(facedict[i])
        cv2.imwrite("images/%s.jpg" %i, facedict[i])
        prediction.append(pred)
        confidence.append(conf)
    output = emotions[max(set(prediction), key=prediction.count)]    
    print("You seem to be %s" %output) 
    facedict.clear()
    return output

# Tkinter GUI for live video capture
def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

# Eel and Flask integration for web GUI
@eel.expose
def getEmotion():
    count = 0
    while True:
        count += 1
        detect_face()
        if args.update:
            update_model(emotions)
            break
        elif count == 10:
            fishface.read("model.xml")
            return identify_emotions()
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options for emotions based music player (Updating the model)")
    parser.add_argument("--update", help="Call for taking new images and retraining the model.", action="store_true")
    args = parser.parse_args()

    # Tkinter setup
    root = tk.Tk()
    root.bind('<Escape>', lambda e: root.quit())
    lmain = tk.Label(root)
    lmain.pack()

    # Start Flask app
    app.run(debug=True)

    # Start Eel app
    eel.start('main.html')

    # Start Tkinter GUI
    show_frame()
    root.mainloop()

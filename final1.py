import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to initialize the webcam
def initialize_camera():
    return cv2.VideoCapture(0)

# Function to capture video frames
def capture_frames(video_capture):
    success, frame = video_capture.read()
    return success, frame

# Function to stop the video capturing
def stop_video_capture(video_capture):
    video_capture.release()

# Function to display the Streamlit app
def main():
  # Center align the title
 st.markdown(
        "<h1 style='text-align: center;'>Welcome to Sign Sense👁️</h1>",
        unsafe_allow_html=True
    )

    # Center align the button
   # st.markdown(
    #    "<div style='text-align: center;'><button>Detect Here</button></div>",
     #   unsafe_allow_html=True
    #)

    # Create a button to start video capturing
if st.markdown(
        "<div style='text-align: center;'><button>Detect Here</button></div>",
        unsafe_allow_html=True
    ):
        cap = initialize_camera()
        detector = HandDetector(maxHands=1)
        classifier = Classifier("C:\\Users\\deepi\\OneDrive\\Desktop\Model\\keras_model.h5", "C:\\Users\\deepi\\OneDrive\\Desktop\\Model\\labels.txt")
        offset = 20
        imgSize = 300
        counter = 0

        labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

        while True:
            success, img = capture_frames(cap)
            if not success:
                break

            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                # Check if imgCrop is not empty
                if not imgCrop.size == 0:
                    imgOutput = img.copy()

                    aspectRatio = h / w
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))

                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    imgWhite[:imgResize.shape[0], :imgResize.shape[1]] = imgResize

                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    text_to_speak = labels[index]

                    # Speak the predicted label
                    engine.say(text_to_speak)
                    engine.runAndWait()

                    cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0),
                                  cv2.FILLED)

                    cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

                    cv2.imshow('Image', imgOutput)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        stop_video_capture(cap)

if __name__ == "__main__":
    main()

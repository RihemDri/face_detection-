
# ---------------------------------------
# Import the necessary packages
# ---------------------------------------
import cv2
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="Face Detection",
    page_icon="👤",
    layout="centered")
# ---------------------------------------
#  : Load the face cascade classifier
# ---------------------------------------

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# ---------------------------------------------------------------------------
#Function to Capture Webcam & Detect Faces
# ---------------------------------------------------------------------------

#This function contains all the steps needed to open the webcam,
#  detect faces, draw rectangles, and show the video live.
def detect_faces():
    # Initialize the webcam
    cap = cv2.VideoCapture(0) #open the default camera /0 → built-in webcam

    while True: #This loop keeps running until the user presses 'q'.

        #--------------------------------
         #Read frame → detect faces → show frame → repeat.
         #--------------------------------

        # Read the frames from the webcam
        ret, frame = cap.read()
        # cap red e turns:
        #ret → True/False (did the camera successfully capture a frame?)
        #frame → the actual image captured from the webcam


        # Convert the frames to grayscale (algorithm requires grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the faces using the face cascade classifier
        #detectMultiScale() finds faces

        #This loads a pre-trained model that knows how to detect faces
        #faces is a list of all faces found.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))


# ---------------------------------------
# Draw rectangles around the detected faces
# ---------------------------------------
        #Each face is represented as:(x, y, w, h) meaning:
                 #x = left coordinate /y = top coordinate
                 # w = width of the face / h = height of the face
        for (x, y, w, h) in faces: #Loops through all detected faces
                    # Draw  GREEN (0, 255, 0) rectangles around the detected faces 
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #2 → thickness of the rectangle borde
            # Draw label "Face" above the rectangle
            cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # Display the frames in a window
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


    # ---------------------------------------
    #streamlit app main function
    # ---------------------------------------
def app():
    st.title("Face Detection App")
    st.write("Click the button to start detecting faces!")

    if st.button("Start"):
        detect_faces() # Call the detect_faces function

if __name__ == "__main__":
    app()


 


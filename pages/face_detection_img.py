#------------------------------------------------------
#1. Import OpenCV
#------------------------------------------------------
import streamlit as st
import numpy as np
from PIL import Image

import cv2
#Loads the OpenCV library, which allows you to work with images and video.
st.title("📸   Face Detection  ")
st.write("Upload an image and the app will detect faces using Haar Cascade.")
#------------------------------------------------------
#2. Load the Haar Cascade Face Detector
#------------------------------------------------------

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
#------------------------------------------------------
 #3. Read Image + Convert to Grayscale
#------------------------------------------------------
# Read the image and convert it to grayscale
#img = cv2.imread('image5.jpg')
#------------------------------------------------------
# Upload image
# #------------------------------------------------------


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to an OpenCV image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

#------------------------------------------------------
#convert the image to grayscale
#------------------------------------------------------
 
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

#------------------------------------------------------
# Detect faces
#------------------------------------------------------
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    st.write(f"🟢 Number of faces detected: {len(faces)}")

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show the result
    st.subheader("Detected Faces")
    st.image(img_array, caption="Detected Faces", use_column_width=True)
st.sidebar.subheader("💡 Tips for Better Detection")
st.sidebar.write("""
1. **Use clear, well-lit photos**
2. **Faces should be front-facing**
3. **Adjust parameters if needed:**
   - Decrease **Scale Factor** if missing faces
   - Increase **Min Neighbors** if too many false detections
   - Adjust **Min Face Size** based on image resolution
4. **For group photos**, try lower Min Neighbors
""")
 
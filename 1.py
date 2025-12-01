import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Face Detection Code Explanation",
    page_icon="👨‍🏫",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2E86AB;
        font-size: 2.8rem;
        margin-bottom: 2rem;
    }
    .section-title {
        color: #2E86AB;
        border-bottom: 3px solid #2E86AB;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    .code-block {
        background-color: #2D3748;
        color: #E2E8F0;
        padding: 15px;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        margin: 15px 0;
        border-left: 5px solid #4ECDC4;
    }
    .explanation-box {
        background-color: #F0F9FF;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #2E86AB;
    }
    .note-box {
        background-color: #FFF3CD;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #FFC107;
    }
    .success-box {
        background-color: #D1E7DD;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #198754;
    }
    .line-highlight {
        background-color: #FFF3CD;
        padding: 3px 8px;
        border-radius: 3px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-title">👨‍🏫 Face Detection Code Step-by-Step Guide</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📚 Learning Modules")
    module = st.sidebar.radio(
        "Choose a module:",
        ["📖 Complete Code Overview", 
         "🔍 Line-by-Line Explanation", 
         "🎮 Interactive Demo",
         "📝 Student Exercise",
         "✅ Quiz & Test"]
    )
    
    if module == "📖 Complete Code Overview":
        show_complete_overview()
    elif module == "🔍 Line-by-Line Explanation":
        show_line_by_line()
    elif module == "🎮 Interactive Demo":
        show_interactive_demo()
    elif module == "📝 Student Exercise":
        show_student_exercise()
    elif module == "✅ Quiz & Test":
        show_quiz()

def show_complete_overview():
    st.markdown('<h2 class="section-title">📖 Complete Code Overview</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome Students! 🎓
    
    In this lesson, we'll explore a complete face detection program. We'll break down each part of the code,
    understand how it works, and learn computer vision concepts along the way.
    """)
    
    # Display the complete code with line numbers
    st.markdown('<h3>🧱 The Complete Face Detection Code</h3>', unsafe_allow_html=True)
    
    complete_code = """import cv2
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Face Detection",
    page_icon="👤",
    layout="centered"
)

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_faces():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()
        
        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect the faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
        
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

def app():
    st.title("Face Detection App")
    st.write("Click the button to start detecting faces!")
    
    if st.button("Start"):
        detect_faces()

if __name__ == "__main__":
    app()"""
    
    st.code(complete_code, language='python', line_numbers=True)
    
    # Code structure breakdown
    st.markdown('<h3>🏗️ Code Structure Breakdown</h3>', unsafe_allow_html=True)
    
    structure_data = {
        "Part": ["Import Libraries", "Page Setup", "Load Model", "Main Function", "Streamlit App", "Program Entry"],
        "Lines": ["1-2", "5-9", "12-14", "17-45", "48-55", "58-59"],
        "Purpose": ["Bring in necessary tools", "Configure web app", "Load face detection model", "Core face detection logic", "Create user interface", "Start the program"]
    }
    
    st.table(structure_data)
    
    # Learning objectives
    st.markdown('<h3>🎯 What You Will Learn</h3>', unsafe_allow_html=True)
    
    objectives = [
        "✅ How to access and use your computer's webcam",
        "✅ How computer vision algorithms detect faces",
        "✅ The difference between color and grayscale images",
        "✅ How to draw shapes and text on images",
        "✅ Creating interactive applications with Streamlit",
        "✅ Proper resource management (opening/closing devices)"
    ]
    
    for objective in objectives:
        st.markdown(objective)

def show_line_by_line():
    st.markdown('<h2 class="section-title">🔍 Line-by-Line Explanation</h2>', unsafe_allow_html=True)
    
    # Create tabs for different sections
    tabs = st.tabs([
        "📦 Imports & Setup", 
        "🤖 Face Detector", 
        "🎥 Webcam Loop", 
        "🎨 Face Detection", 
        "🖼️ Display & Control",
        "🧹 Cleanup"
    ])
    
    with tabs[0]:
        st.markdown("### 📦 Importing Libraries")
        st.markdown('<div class="code-block">import cv2\nimport streamlit as st</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
        st.markdown("""
        #### What these imports do:
        
        **`import cv2`** - OpenCV library
        - The main computer vision library
        - Used for image/video processing, face detection, drawing
        - Contains pre-trained models for face detection
        
        **`import streamlit as st`** - Streamlit framework
        - Creates interactive web applications with Python
        - No HTML/CSS/JavaScript needed
        - Provides buttons, text, images, etc.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="note-box">', unsafe_allow_html=True)
        st.markdown("""
        💡 **Important Note:** 
        If you get an error importing these, install them first:
        ```
        pip install opencv-python streamlit
        ```
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("### 🤖 Loading the Face Detector")
        st.markdown('<div class="code-block">face_cascade = cv2.CascadeClassifier(\n    cv2.data.haarcascades + \'haarcascade_frontalface_default.xml\'\n)</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
            st.markdown("""
            #### This line loads a pre-trained face detector:
            
            **`cv2.CascadeClassifier()`** - Creates a classifier object
            - Takes a path to an XML file containing the trained model
            - The model knows what faces look like
            
            **`cv2.data.haarcascades`** - Built-in OpenCV path
            - OpenCV comes with several pre-trained models
            - This path points to where they're stored
            
            **`haarcascade_frontalface_default.xml`** - The model file
            - Trained to detect front-facing faces
            - Uses the Viola-Jones algorithm
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🎯 Other Available Models:")
            models = [
                "haarcascade_frontalface_default.xml",
                "haarcascade_profileface.xml",
                "haarcascade_eye.xml",
                "haarcascade_smile.xml",
                "haarcascade_fullbody.xml"
            ]
            
            for model in models:
                st.code(model, language='text')
            
            st.markdown('<div class="note-box">', unsafe_allow_html=True)
            st.markdown("💡 Try changing to `haarcascade_eye.xml` to detect eyes instead!")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("### 🎥 Webcam Initialization & Loop")
        st.markdown('<div class="code-block">def detect_faces():\n    cap = cv2.VideoCapture(0)\n    \n    while True:</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
        st.markdown("""
        #### Setting up the webcam and video loop:
        
        **`def detect_faces():`** - Function definition
        - Creates a reusable block of code
        - All face detection logic goes here
        
        **`cap = cv2.VideoCapture(0)`** - Open webcam
        - `0` = First camera (built-in webcam)
        - `1` = Second camera (USB webcam)
        - `cap` is now a "video capture object"
        
        **`while True:`** - Infinite loop
        - Runs continuously until we stop it
        - Each loop iteration processes one video frame
        - Creates real-time video effect
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visual demonstration
        st.markdown("#### 📊 Video Capture Flow")
        st.image("https://www.researchgate.net/profile/Mohamed-Elhoseny-2/publication/344786833/figure/fig3/AS:945790100275202@1602495200429/Webcam-capture-process.png", 
                 caption="Webcam → Capture → Process → Display", use_column_width=True)
    
    with tabs[3]:
        st.markdown("### 🎨 Face Detection Process")
        st.code("""
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        """, language='python')
        
        # Interactive parameter explanation
        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
        st.markdown("""
        #### Step-by-step face detection:
        
        1. **`cap.read()`** - Capture one frame from webcam
           - `ret` = True if successful, False if error
           - `frame` = The actual image data
           
        2. **`cvtColor()`** - Convert to grayscale
           - Face detection works better without color
           - Reduces processing complexity
           
        3. **`detectMultiScale()`** - Find faces
           - Parameters control sensitivity:
           - `scaleFactor=1.3` - How much to scale down image
           - `minNeighbors=5` - How many neighbors each detection should have
           
        4. **Loop through detected faces**
           - Each face has coordinates (x, y, width, height)
           - Draw green rectangle around each face
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive parameter slider
        st.markdown("#### 🎚️ Try Adjusting Parameters:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scale_factor = st.slider("scaleFactor", 1.1, 2.0, 1.3, 0.1)
            st.write(f"**Current: {scale_factor}**")
            st.write("Higher = Faster but may miss faces")
            st.write("Lower = Slower but more accurate")
        
        with col2:
            min_neighbors = st.slider("minNeighbors", 1, 10, 5)
            st.write(f"**Current: {min_neighbors}**")
            st.write("Higher = Fewer detections, better quality")
            st.write("Lower = More detections, more false alarms")
    
    with tabs[4]:
        st.markdown("### 🖼️ Display & User Control")
        st.code("""
cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
        """, language='python')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
            st.markdown("""
            #### Displaying the video:
            
            **`cv2.imshow()`** - Show image in window
            - First parameter: Window title
            - Second parameter: Image to display
            - Updates the window each frame
            
            **`waitKey(1)`** - Check for key press
            - Waits 1 millisecond (very short)
            - Returns key code if pressed
            - Allows window to refresh
            
            **`ord('q')`** - Get 'q' key code
            - Converts 'q' to its ASCII code (113)
            - `& 0xFF` ensures cross-platform compatibility
            
            **`break`** - Exit the loop
            - Stops the `while True` loop
            - Program continues to cleanup code
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### ⌨️ Other Key Controls:")
            st.markdown("""
            You can modify the code to use different keys:
            
            ```python
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Press 's' to save screenshot
            if cv2.waitKey(1) & 0xFF == ord('s'):
                cv2.imwrite('screenshot.jpg', frame)
            
            # Press spacebar to pause
            if cv2.waitKey(1) & 0xFF == ord(' '):
                cv2.waitKey(0)  # Wait indefinitely
            ```
            """)
    
    with tabs[5]:
        st.markdown("### 🧹 Resource Cleanup")
        st.code("""
cap.release()
cv2.destroyAllWindows()
        """, language='python')
        
        st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
        st.markdown("""
        #### Why cleanup is important:
        
        **`cap.release()`** - Release the webcam
        - Closes the connection to the camera
        - Allows other programs to use it
        - Prevents camera from being "locked"
        
        **`cv2.destroyAllWindows()`** - Close all windows
        - Removes OpenCV display windows
        - Frees up memory
        - Prevents window from staying open
        
        #### ⚠️ What happens without cleanup:
        - Camera remains in use even after program ends
        - Might need to restart computer
        - Memory leaks over time
        - Other apps can't access webcam
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("""
        🎯 **Best Practice:** Always release resources!
        Think of it like closing a book after reading or turning off lights when leaving a room.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

def show_interactive_demo():
    st.markdown('<h2 class="section-title">🎮 Interactive Demo</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Try the Face Detection Yourself!
    
    Below you can upload an image and see how the face detection works.
    The code will process your image and show you where faces are detected.
    """)
    
    # Image upload
    uploaded_file = st.file_uploader("📤 Upload an image with faces", 
                                    type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        # Process image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Display original
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Process with face detection
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚙️ Processing Steps:")
            
            steps = [
                ("1️⃣ Load Image", "Read the uploaded image"),
                ("2️⃣ Convert to Grayscale", "Remove color for easier processing"),
                ("3️⃣ Detect Faces", "Use Viola-Jones algorithm"),
                ("4️⃣ Draw Boxes", "Mark detected faces with green rectangles"),
                ("5️⃣ Add Labels", "Put 'Face' text above each detection")
            ]
            
            for step, description in steps:
                with st.expander(step):
                    st.write(description)
        
        with col2:
            # Actually process the image
            with st.spinner("Detecting faces..."):
                # Convert to OpenCV format
                if len(image_np.shape) == 3:
                    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                else:
                    image_cv = image_np
                
                # Load classifier
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                
                # Convert to grayscale
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.3, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                
                # Draw rectangles
                for (x, y, w, h) in faces:
                    cv2.rectangle(image_cv, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(image_cv, 'Face', (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Convert back for display
                result_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                
                # Display result
                st.image(result_image, caption=f"Detected {len(faces)} face(s)", 
                        use_column_width=True)
                
                if len(faces) > 0:
                    st.success(f"🎉 Successfully detected {len(faces)} face(s)!")
                else:
                    st.warning("🤔 No faces detected. Try a different image or adjust parameters.")
        
        # Show detection details
        if len(faces) > 0:
            st.markdown("#### 📊 Detection Details:")
            
            for i, (x, y, w, h) in enumerate(faces, 1):
                st.write(f"**Face {i}:** Position: ({x}, {y}), Size: {w}×{h} pixels")
        
        # Parameter adjustment
        st.markdown("---")
        st.markdown("#### 🎚️ Adjust Detection Parameters:")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            new_scale = st.slider("Scale Factor", 1.1, 2.0, 1.3, 0.1)
        
        with col_b:
            new_neighbors = st.slider("Min Neighbors", 1, 10, 5)
        
        with col_c:
            new_min_size = st.slider("Min Face Size", 20, 100, 30)
        
        if st.button("🔄 Re-detect with new parameters"):
            with st.spinner("Re-detecting..."):
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=new_scale, 
                    minNeighbors=new_neighbors, 
                    minSize=(new_min_size, new_min_size)
                )
                
                # Create new result
                new_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) if len(image_np.shape) == 3 else image_np
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(new_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(new_image, 'Face', (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                new_result = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
                st.image(new_result, caption=f"New detection: {len(faces)} faces", 
                        use_column_width=True)

def show_student_exercise():
    st.markdown('<h2 class="section-title">📝 Student Exercise</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Practice What You've Learned!
    
    Below are exercises to test your understanding.
    Try to complete them without looking at the solutions first!
    """)
    
    # Exercise tabs
    ex_tabs = st.tabs(["Exercise 1", "Exercise 2", "Exercise 3", "Solutions"])
    
    with ex_tabs[0]:
        st.markdown("#### 🎯 Exercise 1: Modify the Code")
        st.markdown("""
        **Task:** Modify the face detection code to:
        1. Change the rectangle color from green to blue
        2. Make the rectangle thicker
        3. Change the text from "Face" to "Person"
        4. Use a different font for the text
        
        **Starting Code:**
        ```python
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        ```
        
        **Hints:**
        - Blue color in BGR format is `(255, 0, 0)`
        - Thickness is the last number in `rectangle()`
        - Try `FONT_HERSHEY_COMPLEX` or `FONT_HERSHEY_DUPLEX`
        """)
        
        # Interactive code editor
        st.markdown("**Try it here:**")
        exercise1_code = st.text_area("Write your modified code:", 
                                     height=150,
                                     value="""for (x, y, w, h) in faces:
    # Modify these lines:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)""")
    
    with ex_tabs[1]:
        st.markdown("#### 🎯 Exercise 2: Add New Feature")
        st.markdown("""
        **Task:** Add a feature to count and display the number of faces detected.
        
        Requirements:
        1. Count how many faces are detected
        2. Display the count on the video feed
        3. Put the count at the top-left corner of the screen
        4. Make the text red and larger than the face labels
        
        **Example Output:**
        ```
        Faces detected: 3
        ```
        
        **Hints:**
        - Use `len(faces)` to get the count
        - Use `cv2.putText()` to display text
        - Top-left corner is position `(10, 30)` for example
        - Red color is `(0, 0, 255)` in BGR
        """)
        
        # Solution input
        st.text_area("Write your solution:", height=150,
                    placeholder="Add your code here...")
    
    with ex_tabs[2]:
        st.markdown("#### 🎯 Exercise 3: Error Handling")
        st.markdown("""
        **Task:** Add error handling to make the code more robust.
        
        Current issues with the code:
        1. What if the webcam fails to open?
        2. What if no frame is captured?
        3. What if the classifier file is missing?
        
        **Add error handling for:**
        ```python
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            # Handle error
        ```
        
        ```python
        ret, frame = cap.read()
        if not ret:
            # Handle error
        ```
        
        ```python
        if face_cascade.empty():
            # Handle error
        ```
        
        **Hints:**
        - Use `if` statements to check conditions
        - Use `st.error()` to show error messages in Streamlit
        - Use `return` to exit the function early on error
        """)
        
        st.text_area("Write your error handling code:", height=200,
                    placeholder="Add error handling code here...")
    
    with ex_tabs[3]:
        st.markdown("#### ✅ Solutions")
        
        st.markdown("**Exercise 1 Solution:**")
        st.code("""for (x, y, w, h) in faces:
    # Changed color to blue, increased thickness to 4
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
    # Changed text to 'Person', using different font
    cv2.putText(frame, 'Person', (x, y - 10), 
               cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 0), 2)""", language='python')
        
        st.markdown("**Exercise 2 Solution:**")
        st.code("""# Count faces
face_count = len(faces)

# Display count at top-left corner
cv2.putText(frame, f'Faces detected: {face_count}', 
           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Rest of the face drawing code...
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, 'Face', (x, y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)""", language='python')
        
        st.markdown("**Exercise 3 Solution:**")
        st.code("""def detect_faces():
    # Check if classifier loaded
    if face_cascade.empty():
        st.error("Failed to load face detector!")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        st.error("Cannot open webcam!")
        return
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        # Check if frame was captured
        if not ret:
            st.error("Failed to capture frame!")
            break
        
        # Rest of the code...
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow('Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()""", language='python')

def show_quiz():
    st.markdown('<h2 class="section-title">✅ Quiz & Knowledge Check</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Test Your Understanding! 📝
    
    Answer these questions to check what you've learned.
    Select the correct answers for each question.
    """)
    
    # Quiz questions
    quiz_questions = [
        {
            "question": "What does `cv2.VideoCapture(0)` do?",
            "options": [
                "Captures video from the first webcam",
                "Takes a screenshot",
                "Records audio",
                "Opens a video file"
            ],
            "correct": 0
        },
        {
            "question": "Why do we convert images to grayscale for face detection?",
            "options": [
                "Color images are too large",
                "The algorithm works better with contrast",
                "Grayscale looks better",
                "It's required by law"
            ],
            "correct": 1
        },
        {
            "question": "What does the `scaleFactor` parameter control?",
            "options": [
                "How fast the video plays",
                "How much to reduce image size during detection",
                "The color scale of the image",
                "The size of the display window"
            ],
            "correct": 1
        },
        {
            "question": "What does `cap.release()` do?",
            "options": [
                "Starts recording video",
                "Releases the webcam for other programs",
                "Saves the video to a file",
                "Increases video quality"
            ],
            "correct": 1
        },
        {
            "question": "Which key do we press to quit the face detection program?",
            "options": [
                "'q' key",
                "Spacebar",
                "Escape key",
                "Enter key"
            ],
            "correct": 0
        }
    ]
    
    # Display quiz
    score = 0
    user_answers = []
    
    for i, q in enumerate(quiz_questions, 1):
        st.markdown(f"#### Question {i}: {q['question']}")
        
        # Get user answer
        user_answer = st.radio(
            f"Select your answer for question {i}:",
            q['options'],
            key=f"q{i}"
        )
        
        user_answers.append(user_answer)
        
        # Check answer on button press
        if st.button(f"Check Answer {i}", key=f"check{i}"):
            if q['options'].index(user_answer) == q['correct']:
                st.success("✅ Correct!")
                score += 1
            else:
                st.error(f"❌ Incorrect. The correct answer is: {q['options'][q['correct']]}")
        
        st.markdown("---")
    
    # Calculate and display final score
    if st.button("📊 Calculate Final Score"):
        st.markdown(f"### Your Score: {score}/{len(quiz_questions)}")
        
        if score == len(quiz_questions):
            st.balloons()
            st.success("🎉 Excellent! You've mastered the basics!")
        elif score >= len(quiz_questions) // 2:
            st.info("👍 Good job! Review the sections you missed.")
        else:
            st.warning("📚 Keep learning! Review the material and try again.")
    
    # Learning resources
    st.markdown("---")
    st.markdown("#### 📚 Additional Learning Resources")
    
    resources = [
        "📖 [OpenCV Documentation](https://docs.opencv.org/)",
        "🎥 [Computer Vision Tutorials on YouTube](https://www.youtube.com/results?search_query=opencv+tutorial)",
        "💻 [Practice coding on Kaggle](https://www.kaggle.com/learn/computer-vision)",
        "📝 [Streamlit Documentation](https://docs.streamlit.io/)",
        "👨‍🏫 [Course: Introduction to Computer Vision](https://www.coursera.org/learn/introduction-computer-vision)"
    ]
    
    for resource in resources:
        st.markdown(resource)

if __name__ == "__main__":
    main()
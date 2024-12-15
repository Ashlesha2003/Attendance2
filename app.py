import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from PIL import Image

# Ensure the dataset and attendance directories exist
DATASET_DIR = "dataset"
ATTENDANCE_FILE = "attendance.csv"
os.makedirs(DATASET_DIR, exist_ok=True)

# Load Haar Cascade Classifier for face detection
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_face_cascade()

# Ensure attendance file exists
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Roll Number", "Name", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

def detect_and_crop_faces(uploaded_file):
    """Detect and crop faces from an uploaded image."""
    # Read the image using PIL
    image = Image.open(uploaded_file)
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    elif len(img_array.shape) == 2:
        gray = img_array
    else:
        st.error("Unsupported image format")
        return []
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    cropped_faces = []
    for (x, y, w, h) in faces:
        # Crop the face
        face_img = gray[y:y+h, x:x+w]
        cropped_faces.append(face_img)
    
    return cropped_faces

def compute_histogram(image):
    """Compute histogram of the image for comparison."""
    # Resize to a standard size
    resized = cv2.resize(image, (100, 100))
    # Compute histogram
    hist = cv2.calcHist([resized], [0], None, [256], [0, 256]).flatten()
    return hist

def compare_faces(face1, face2, threshold=0.7):
    """Compare two faces using histogram comparison."""
    hist1 = compute_histogram(face1)
    hist2 = compute_histogram(face2)
    
    # Compare histograms
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity > threshold

def encode_faces(data_dir):
    """Encode faces for recognition."""
    faces = []
    labels = []
    label_dict = {}
    label_count = 0

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(("jpg", "png", "jpeg")):
                img_path = os.path.join(root, file)
                label = os.path.basename(root)

                # Assign a new label for each student
                if label not in label_dict:
                    label_dict[label] = label_count
                    label_count += 1

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                # Detect faces
                faces_rects = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces_rects:
                    face_img = img[y:y+h, x:x+w]
                    faces.append(face_img)
                    labels.append(label_dict[label])
    
    return faces, labels, label_dict

def mark_attendance(name):
    """Mark attendance for a student."""
    df = pd.read_csv(ATTENDANCE_FILE)
    if name not in df["Name"].values:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        roll_number, student_name = name.split("_")
        df.loc[len(df)] = [roll_number, student_name, timestamp]
        df.to_csv(ATTENDANCE_FILE, index=False)
        st.success(f"Attendance marked for {name}")
    else:
        st.warning(f"{name} already marked present.")

def add_student_page():
    """Streamlit page for adding a new student."""
    st.header("Add New Student")
    
    # Input student details
    name = st.text_input("Student Name")
    roll_number = st.text_input("Roll Number")
    
    # File uploader for student images
    uploaded_files = st.file_uploader(
        "Upload Student Images (multiple allowed)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    
    if st.button("Save Student Data"):
        if not name or not roll_number:
            st.error("Please fill in all details")
            return
        
        if not uploaded_files:
            st.error("Please upload at least one image")
            return
        
        # Create student folder
        student_folder = os.path.join(DATASET_DIR, f"{roll_number}_{name}")
        os.makedirs(student_folder, exist_ok=True)
        
        # Save uploaded images
        for i, uploaded_file in enumerate(uploaded_files):
            # Detect and save faces from the uploaded image
            faces = detect_and_crop_faces(uploaded_file)
            
            if faces:
                # Save the first detected face
                face_path = os.path.join(student_folder, f"{name}_{i+1}.jpg")
                cv2.imwrite(face_path, faces[0])
                st.success(f"Image {i+1} saved successfully")
            else:
                st.warning(f"No face detected in image {i+1}")

def take_attendance_page():
    """Streamlit page for taking attendance."""
    st.header("Take Attendance")
    
    # Try to encode faces
    try:
        faces, labels, label_dict = encode_faces(DATASET_DIR)
        
        if not faces:
            st.error("No faces found in dataset. Please add students first.")
            return
    except Exception as e:
        st.error(f"Error encoding faces: {e}")
        return
    
    # File uploader for attendance images
    uploaded_file = st.file_uploader(
        "Upload Image for Attendance", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image")
        
        # Read image
        img = cv2.imread(uploaded_file.name, cv2.IMREAD_GRAYSCALE)
        
        # Detect faces in the uploaded image
        faces_in_image = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Recognize faces
        recognized_students = []
        for (x, y, w, h) in faces_in_image:
            # Extract face region
            face_img = img[y:y+h, x:x+w]
            
            # Compare with known faces
            for known_face, label in zip(faces, labels):
                if compare_faces(known_face, face_img):
                    # Find the name corresponding to this label
                    name = [k for k, v in label_dict.items() if v == label][0]
                    recognized_students.append(name)
                    break
        
        # Mark attendance for recognized students
        if recognized_students:
            for student in set(recognized_students):
                mark_attendance(student)
        else:
            st.warning("No students recognized in the image")

def view_attendance_page():
    """Streamlit page for viewing attendance."""
    st.header("Attendance Record")
    
    # Read and display attendance
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error reading attendance file: {e}")

def main():
    """Main Streamlit application."""
    st.title("Face Recognition Attendance System")
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Page", 
        ["Add Student", "Take Attendance", "View Attendance"]
    )
    
    # Page routing
    if page == "Add Student":
        add_student_page()
    elif page == "Take Attendance":
        take_attendance_page()
    else:
        view_attendance_page()

# Requirements for Streamlit deployment
if _name_ == "_main_":
    main()
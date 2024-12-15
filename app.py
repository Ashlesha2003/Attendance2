import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from PIL import Image
import face_recognition

# Ensure the dataset and attendance directories exist
DATASET_DIR = "dataset"
ATTENDANCE_FILE = "attendance.csv"
os.makedirs(DATASET_DIR, exist_ok=True)

# Ensure attendance file exists
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Roll Number", "Name", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

def detect_and_crop_faces(uploaded_file):
    """Detect and crop faces from an uploaded image."""
    # Read the image using PIL
    image = Image.open(uploaded_file)
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Detect faces
    face_locations = face_recognition.face_locations(img_array)
    cropped_faces = []
    
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_img = img_array[top:bottom, left:right]
        cropped_faces.append(face_img)
    
    return cropped_faces

def encode_faces(data_dir):
    """Encode faces for recognition."""
    known_face_encodings = []
    known_face_names = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(("jpg", "png", "jpeg")):
                img_path = os.path.join(root, file)
                label = os.path.basename(root)
                
                # Read image
                image = face_recognition.load_image_file(img_path)
                
                # Encode faces
                face_encodings = face_recognition.face_encodings(image)
                
                for encoding in face_encodings:
                    known_face_encodings.append(encoding)
                    known_face_names.append(label)
    
    return known_face_encodings, known_face_names

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
                Image.fromarray(faces[0]).save(face_path)
                st.success(f"Image {i+1} saved successfully")
            else:
                st.warning(f"No face detected in image {i+1}")

def take_attendance_page():
    """Streamlit page for taking attendance."""
    st.header("Take Attendance")
    
    # Encode faces from dataset
    try:
        known_face_encodings, known_face_names = encode_faces(DATASET_DIR)
        
        if not known_face_encodings:
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
        img = face_recognition.load_image_file(uploaded_file)
        
        # Find face locations and encodings in the uploaded image
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        
        # Recognize faces
        recognized_students = []
        for face_encoding in face_encodings:
            # Compare face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            
            # Find the best match
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                recognized_students.append(name)
        
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
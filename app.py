import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from PIL import Image

DATASET_DIR = "dataset"
ATTENDANCE_FILE = "attendance.csv"
os.makedirs(DATASET_DIR, exist_ok=True)


@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_cascade = load_face_cascade()

# Create attendance file if not exists
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Roll Number", "Name", "Time"]).to_csv(ATTENDANCE_FILE, index=False)


def detect_and_crop_faces(uploaded_file):
    """Detect and crop faces from an uploaded image."""
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    elif len(img_array.shape) == 2:
        gray = img_array
    else:
        st.error("Unsupported image format")
        return []

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    cropped_faces = []
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        # ✅ Resize to consistent size
        face_img = cv2.resize(face_img, (200, 200))
        cropped_faces.append(face_img)
    return cropped_faces


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

                if label not in label_dict:
                    label_dict[label] = label_count
                    label_count += 1

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                faces_rects = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces_rects:
                    face_img = img[y:y+h, x:x+w]
                    # ✅ Resize to consistent size
                    face_img = cv2.resize(face_img, (200, 200))
                    faces.append(face_img)
                    labels.append(label_dict[label])
    return faces, labels, label_dict


def train_face_recognizer():
    """Train the face recognizer."""
    st.info("Encoding faces from the dataset...")
    faces, labels, label_dict = encode_faces(DATASET_DIR)

    if not faces:
        st.error("No faces found in dataset.")
        return None, label_dict

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    st.success("Training complete.")
    return recognizer, label_dict


def mark_attendance(name):
    """Mark attendance for a student."""
    df = pd.read_csv(ATTENDANCE_FILE)
    if name not in df["Name"].values:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        roll_number, student_name = name.split("_")
        df.loc[len(df)] = [roll_number, student_name, timestamp]
        df.to_csv(ATTENDANCE_FILE, index=False)
        st.success(f"Attendance marked for {student_name} ({roll_number})")
    else:
        st.warning(f"{name} already marked present.")


def add_student_page():
    """Streamlit page for adding a new student."""
    st.header("Add New Student")

    name = st.text_input("Student Name")
    roll_number = st.text_input("Roll Number")

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

        student_folder = os.path.join(DATASET_DIR, f"{roll_number}_{name}")
        os.makedirs(student_folder, exist_ok=True)

        for i, uploaded_file in enumerate(uploaded_files):
            faces = detect_and_crop_faces(uploaded_file)
            if faces:
                for j, face in enumerate(faces):  # ✅ Save all detected faces
                    face_path = os.path.join(student_folder, f"{name}_{i+1}_{j+1}.jpg")
                    cv2.imwrite(face_path, face)
                st.success(f"Image {i+1}: Saved {len(faces)} face(s) successfully")
            else:
                st.warning(f"No face detected in image {i+1}")


def take_attendance_page():
    """Streamlit page for taking attendance."""
    st.header("Take Attendance")

    recognizer, label_dict = train_face_recognizer()
    if recognizer is None:
        return

    uploaded_file = st.file_uploader(
        "Upload Image for Attendance",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image")

        img = Image.open(uploaded_file)
        img_array = np.array(img)

        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        elif len(img_array.shape) == 2:
            gray = img_array
        else:
            st.error("Unsupported image format")
            return

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        recognized_students = []
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))  # ✅ Resize before recognition

            try:
                label, confidence = recognizer.predict(face_img)
                st.write(f"Detected Label: {label}, Confidence: {confidence:.2f}")

                if confidence < 50:  # ✅ Stricter threshold
                    name = [k for k, v in label_dict.items() if v == label][0]
                    recognized_students.append(name)
                else:
                    st.warning("Face detected but not confident enough to recognize.")
            except Exception as e:
                st.error(f"Error in face recognition: {e}")

        if recognized_students:
            for student in set(recognized_students):
                mark_attendance(student)
        else:
            st.warning("No students recognized in the image.")


def view_attendance_page():
    """Streamlit page for viewing attendance."""
    st.header("Attendance Record")

    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error reading attendance file: {e}")


def main():
    """Main Streamlit application."""
    st.title("Smart Attendance System")

    page = st.sidebar.selectbox(
        "Select Page",
        ["Add Student", "Take Attendance", "View Attendance"]
    )

    if page == "Add Student":
        add_student_page()
    elif page == "Take Attendance":
        take_attendance_page()
    else:
        view_attendance_page()


if __name__ == "__main__":
    main()
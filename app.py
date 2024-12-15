import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

# Load Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Folder paths
dataset_dir = "dataset"
attendance_file = "attendance.csv"

# Ensure necessary folders and files exist
os.makedirs(dataset_dir, exist_ok=True)
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Roll Number", "Name", "Time"]).to_csv(attendance_file, index=False)

# Function to encode faces using LBPH face recognizer
def encode_faces(data_dir):
    faces = []
    labels = []
    label_dict = {}
    label_count = 0

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(("jpg", "png")):
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

# Function to add a new student
def add_student():
    name = input("Enter student's name: ")
    roll_number = input("Enter student's roll number: ")
    student_folder = os.path.join(dataset_dir, f"{roll_number}_{name}")
    os.makedirs(student_folder, exist_ok=True)

    print("Capturing photos for the student. Press 'c' to capture a photo, or 'q' to finish.")
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Add Student - Press 'c' to Capture Photo", frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):  # Capture photo
            if len(faces) > 0:
                x, y, w, h = faces[0]
                img_path = os.path.join(student_folder, f"{name}_{count + 1}.jpg")
                cv2.imwrite(img_path, frame[y:y+h, x:x+w])
                print(f"Photo {count + 1} captured and saved.")
                count += 1
            else:
                print("No face detected. Please try again.")
        elif key & 0xFF == ord('q'):  # Quit capturing
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data for {name} (Roll Number: {roll_number}) has been saved.")

# Function to mark attendance
def mark_attendance(name):
    df = pd.read_csv(attendance_file)
    if name not in df["Name"].values:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        roll_number, student_name = name.split("_")
        df.loc[len(df)] = [roll_number, student_name, timestamp]
        df.to_csv(attendance_file, index=False)
        print(f"Attendance marked for {name}")
    else:
        print(f"{name} already marked present.")

# Function to train a face recognizer (LBPH)
def train_face_recognizer():
    print("Encoding faces from the dataset...")
    faces, labels, label_dict = encode_faces(dataset_dir)
    
    if not faces:
        print("No faces found in dataset.")
        return None, label_dict

    # Initialize the LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the recognizer on the faces and labels
    recognizer.train(faces, np.array(labels))

    print("Training complete.")
    return recognizer, label_dict
# Function to take attendance
def take_attendance():
    recognizer, label_dict = train_face_recognizer()
    
    if recognizer is None:
        return

    cap = cv2.VideoCapture(0)
    print("Starting attendance system. Press 'q' to quit.")
    
    marked_students = set()  # Keep track of students whose attendance has been marked

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract face region
            face_img = gray[y:y+h, x:x+w]
            
            # Recognize the face using LBPH
            label, confidence = recognizer.predict(face_img)

            if confidence < 100:  # Lower confidence is better
                name = [k for k, v in label_dict.items() if v == label][0]
                
                # Check if this student's attendance has been marked
                if name not in marked_students:
                    mark_attendance(name)
                    marked_students.add(name)  # Mark this student as attended

            # Draw rectangle around the face without showing name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the camera feed
        cv2.imshow("Attendance System", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Function to view attendance
def view_attendance():
    df = pd.read_csv(attendance_file)
    print("\nAttendance Record:")
    print(df)

# Main menu
def main():
    while True:
        print("\nChoose an option:")
        print("1. Add New Student")
        print("2. Take Attendance")
        print("3. View Attendance")
        print("4. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            add_student()
        elif choice == "2":
            take_attendance()
        elif choice == "3":
            view_attendance()
        elif choice == "4":
            print("Exiting the system. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

import os
import cv2
import numpy as np

class FaceRecognition:
    def __init__(self):
        self.IMAGES_FOLDER = 'images'
        self.DATASET_FOLDER = 'dataset'
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        os.makedirs(self.IMAGES_FOLDER, exist_ok=True)
        os.makedirs(self.DATASET_FOLDER, exist_ok=True)
        self.database = Database()

    def train_faces(self):
        images_dir = self.IMAGES_FOLDER
        dataset_dir = self.DATASET_FOLDER

        if not os.path.exists(images_dir):
            print(f"Images directory {images_dir} does not exist.")
            return False

        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

        if not image_files:
            print(f"No images found in {images_dir}. Training aborted.")
            return False

        face_samples = []
        ids = []

        for image_file in image_files:
            image_path = os.path.join(images_dir, image_file)
            img = cv2.imread(image_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Extract ID from the filename (format: id.number.jpg)
            id_str = image_file.split('.')[0]
            if not id_str.isdigit():
                print(f"Skipping {image_file}: invalid ID format.")
                continue
            id_int = int(id_str)

            faces = self.face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

            for (x, y, w, h) in faces:
                face_samples.append(gray_img[y:y+h, x:x+w])
                ids.append(id_int)

        if not face_samples:
            print("No faces found in any images. Training aborted.")
            return False

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(face_samples, np.array(ids))

        trainer_path = os.path.join(dataset_dir, 'trainer.yml')
        recognizer.save(trainer_path)
        print(f"Training completed successfully. Model saved to {trainer_path}")
        return True

class Database:
    def add_student(self, name, contact, student_id, image_filenames):
        print(f"Adding student: {name}, ID: {student_id}, Images: {image_filenames}")

# Initialize for use in Dash app
face_recognition = FaceRecognition()
import unittest
import os
import face_recognition  # Your own module

class TestFaceModel(unittest.TestCase):

    def setUp(self):
        self.model_path = "dataset/trainer.yml"

        # Clean previous model (optional)
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

    def test_train_faces_creates_model_file(self):
        face_recognition.train_faces()
        self.assertTrue(os.path.exists(self.model_path), "trainer.yml was not created.")

    def tearDown(self):
        if os.path.exists(self.model_path):
            os.remove(self.model_path)

if __name__ == "__main__":
    unittest.main()

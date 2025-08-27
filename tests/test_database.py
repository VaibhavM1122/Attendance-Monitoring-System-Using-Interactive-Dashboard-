import unittest
import database

class TestDatabase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.admin_username = "unittest_admin"
        cls.admin_password = "test123"
        cls.roll_number = "UT001"
        cls.student_data = {
            "name": "Unit Test Student",
            "contact": "9876543210",
            "roll_number": cls.roll_number,
            "face_image": "unittest_face.jpg"
        }

    def setUp(self):
        database.delete_student(self.roll_number)
        database.admins_collection.delete_one({"username": self.admin_username})

    def test_add_and_authenticate_admin(self):
        result = database.add_admin(self.admin_username, self.admin_password)
        self.assertIn(result, [True, False])  # False if already exists

        auth = database.authenticate_admin(self.admin_username, self.admin_password)
        self.assertTrue(auth)

    def test_add_and_get_student(self):
        added = database.add_student(
            self.student_data["name"],
            self.student_data["contact"],
            self.student_data["roll_number"],
            self.student_data["face_image"]
        )
        self.assertTrue(added)

        student = database.get_student_by_roll(self.roll_number)
        self.assertIsNotNone(student)
        self.assertEqual(student["name"], self.student_data["name"])
        self.assertEqual(student["contact"], self.student_data["contact"])
        self.assertEqual(student["roll_number"], self.roll_number)

    def test_update_student(self):
        database.add_student(**self.student_data)

        updated = database.update_student(self.roll_number, name="Updated Name", contact="0000000000")
        self.assertTrue(updated)

        student = database.get_student_by_roll(self.roll_number)
        self.assertEqual(student["name"], "Updated Name")
        self.assertEqual(student["contact"], "0000000000")

    def test_update_student_details(self):
        database.add_student(**self.student_data)

        updated = database.update_student_details(self.roll_number, "Another Name", "1111111111")
        self.assertTrue(updated)

        student = database.get_student_by_roll(self.roll_number)
        self.assertEqual(student["name"], "Another Name")
        self.assertEqual(student["contact"], "1111111111")

    def test_delete_student(self):
        database.add_student(**self.student_data)
        deleted = database.delete_student(self.roll_number)
        self.assertTrue(deleted)

        student = database.get_student_by_roll(self.roll_number)
        self.assertIsNone(student)

    def tearDown(self):
        database.delete_student(self.roll_number)
        database.admins_collection.delete_one({"username": self.admin_username})

if __name__ == "__main__":
    unittest.main()

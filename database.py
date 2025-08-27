import pymongo
import sqlite3

# ---------------- MongoDB Connection ----------------

try:
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["FaceRecognitionDB"]
    students_collection = db["students"]
    print("✅ MongoDB connected successfully.")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")
    students_collection = None

# ---------------- Student Operations ----------------

def add_student(name, contact, roll_number, face_image):
    if students_collection is None:
        print("❌ MongoDB not connected.")
        return False

    student_data = {
        "name": name,
        "contact": contact,
        "roll_number": roll_number,
        "face_image": face_image
    }
    students_collection.insert_one(student_data)
    return True

# --

students_col = db.students
def update_student_recognition_time(roll_number, datetime_str):
    result = students_col.update_one(
        {"roll_number": roll_number},
        {"$set": {"last_recognition": datetime_str}}
    )
    return result.modified_count > 0
# --


def get_students():
    if students_collection is None:
        return []
    return list(students_collection.find({}, {"_id": 0}))

def get_student_by_roll(roll_number):
    if students_collection is None:
        return None
    return students_collection.find_one({"roll_number": roll_number}, {"_id": 0})

def delete_student(roll_number):
    if students_collection is None:
        return False
    result = students_collection.delete_one({"roll_number": roll_number})
    return result.deleted_count > 0

def update_student(roll_number, name=None, contact=None):
    if students_collection is None:
        return False
    update_fields = {}
    if name:
        update_fields["name"] = name
    if contact:
        update_fields["contact"] = contact
    if update_fields:
        result = students_collection.update_one(
            {"roll_number": roll_number}, {"$set": update_fields}
        )
        return result.modified_count > 0
    return False

def update_student_details(roll_number, name, contact):
    return update_student(roll_number, name, contact)

# Add this function in your database module (database.py)
def delete_test_student():
    conn = sqlite3.connect('students.db')
    c = conn.cursor()
    c.execute("DELETE FROM students WHERE name = 'Test Student'")
    conn.commit()
    conn.close()

# ---------------- Admin Authentication ----------------

def authenticate_admin(username, password):
    return username == "admin" and password == "admin"
def login_admin(username, password):
    return username == "admin" and password == "admin"


# Real-Time Attendance Monitoring System (Face Recognition + Interactive Dashboard)

A complete, end-to-end **real-time attendance system** that uses **face recognition** for automatic attendance marking and a **Dash web dashboard** for registration, tracking, management, and CSV export. Built with **Python, OpenCV (LBPH), Dash, and SQLite**.

---

## ✨ Key Features
- **Face Recognition Attendance**: Live webcam capture + LBPH model for real-time identification.
- **Interactive Dashboard**: Register students, take attendance, manage records, export CSVs.
- **Auto-Training**: Model retrains automatically after student registration.
- **Duplicate Prevention**: Ensures only one attendance mark per student per day.
- **Search & Manage**: View, search, and delete students (with image cleanup).
- **CSV Export**: Filter by date range and branch, then download.

---

## 🧱 Tech Stack
- **Frontend/Dashboard**: Dash, Plotly, dash-bootstrap-components  
- **Backend/ML**: Python, OpenCV (opencv-contrib), Haar Cascade + LBPH  
- **Database**: SQLite  
- **Data/Utils**: pandas, numpy, Pillow  

---

## 📁 Project Structure


---


## ⚙️ Requirements
Create a `requirements.txt` with:
-  dash
- dash-bootstrap-components
- plotly
- opencv-contrib-python
- numpy
- pandas
- pillow

## 🚀 Getting Started

- 1) Clone the Repo
- 2) Create & activate venv (recommended)
- 3) Install dependencies
- 4) Run the app

## 🔐 Login

- Admin credentials (default):
- Change these after first login (recommended).

## 🧭 How to Use
A) Register Student
- Open Register Student page.
- Enter Name, Branch, Roll Number.
- Capture ~10 images via webcam (system handles the loop).
- On save, the system auto-trains the LBPH model and stores it at dataset/trainer.yml.

B) Take Attendance
- Open Take Attendance page.
- Start camera; recognized faces are matched to student IDs.
- Attendance is inserted once per student per day (status “Present” by default).

C) Manage Students
- View/search students, delete students (removes their images and related records).

D) Export Attendance
- Choose date range and branch → Export CSV (download + saved on server).
  
  ## 📬 Contact

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/vaibhavm1122/)  
[![Email](https://img.shields.io/badge/Email-Contact-red?style=flat&logo=gmail)](mailto:mahaleva0012@gmail.com.com)  
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/VaibhavM1122)


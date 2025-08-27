import io
from PIL import Image
import sqlite3
import plotly.express as px
from datetime import datetime, timedelta
import cv2
import time
import base64
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import os
from face_recognition import face_recognition as my_fr

# Initialize Dash app
# suppress_callback_exceptions=True is CRITICAL for multi-page apps.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
server = app.server

# --- Global Webcam Management ---
webcam_object = None
RECOGNIZER_LBPH = None
CAPTURING_IN_PROGRESS = False
TARGET_IMAGE_COUNT = 10

def get_webcam():
    global webcam_object
    if webcam_object is None or not webcam_object.isOpened():
        webcam_object = cv2.VideoCapture(0)
        if not webcam_object.isOpened():
            print("Error: Could not open webcam after trying.")
            webcam_object = None
    return webcam_object

def release_webcam():
    global webcam_object, CAPTURING_IN_PROGRESS
    if webcam_object is not None:
        webcam_object.release()
        webcam_object = None
        CAPTURING_IN_PROGRESS = False
        print("Webcam released.")

def load_lbph_recognizer():
    global RECOGNIZER_LBPH
    base_dir_my_fr = os.getcwd() # Default
    is_mock_active = 'MockMyFr' in str(type(my_fr))

    if not is_mock_active and hasattr(my_fr, '__file__') and my_fr.__file__ is not None:
        base_dir_my_fr = os.path.dirname(os.path.abspath(my_fr.__file__))

    model_path = os.path.join(base_dir_my_fr, my_fr.DATASET_FOLDER, 'trainer.yml')

    if is_mock_active:
        if os.path.exists(model_path):
            print(f"[MockLoad] Mock trainer.yml found at {model_path}. Setting mock recognizer state.")
            RECOGNIZER_LBPH = "MockRecognizerLoaded"
            return True
        else:
            print(f"[MockLoad] Mock trainer.yml NOT found at {model_path}.")
            RECOGNIZER_LBPH = None
            return False

    if os.path.exists(model_path):
        try:
            RECOGNIZER_LBPH = cv2.face.LBPHFaceRecognizer_create()
            RECOGNIZER_LBPH.read(model_path)
            print(f"REAL Recognizer loaded successfully from {model_path}.")
            return True
        except Exception as e:
            print(f"Error loading REAL recognizer from {model_path}: {e}")
            RECOGNIZER_LBPH = None
            return False
    else:
        print(f"REAL Trainer.yml not found at {model_path}. Please train the model.")
        RECOGNIZER_LBPH = None
        return False

def init_db():
    conn = sqlite3.connect('attendance_app.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, branch TEXT NOT NULL,
                student_id TEXT UNIQUE NOT NULL, registration_date TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT NOT NULL, date TEXT NOT NULL,
                time TEXT NOT NULL, status TEXT DEFAULT 'Present',
                FOREIGN KEY (student_id) REFERENCES students(student_id))''')
    c.execute('''CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL)''')
    if not c.execute("SELECT * FROM admins WHERE username='admin'").fetchone():
        c.execute("INSERT INTO admins (username, password) VALUES (?, ?)", ('admin', 'admin123'))
    conn.commit()
    conn.close()
    print("Dash app database 'attendance_app.db' initialized.")

init_db()

login_page = html.Div([
    dbc.Row([
        dbc.Col(md=4),
        dbc.Col([
            html.Div([
                html.H1("Face Attendance System (S.N.J.B) ", className="text-center mb-4"),
                html.Img(src=app.get_asset_url('login_icon.png') if os.path.exists('assets/login_icon.png') else "https://via.placeholder.com/150.png?text=Login",
                         className="rounded-circle mx-auto d-block mb-4", style={'width': '150px'}),
                dbc.Input(id='username', placeholder="Username", className="mb-3"),
                dbc.Input(id='password', type="password", placeholder="Password", className="mb-3"),
                dbc.Button("Login", id='login-button', color="primary", className="w-100 mb-3"),
                html.Div(id='login-message', className="text-center")
            ], className="p-4 border rounded shadow", style={'backgroundColor': '#2c3e50'})
        ], md=4),
        dbc.Col(md=4)
    ], justify="center", className="py-5 align-items-center",
     style={'background': 'linear-gradient(135deg, #1a2a6c, #b21f1f, #1a2a6c)', 'minHeight': '100vh'})
])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='session', data={'logged_in': False}),
    dcc.Store(id='captured-image-count-store', data={'student_id': None, 'count': 0, 'filenames': []}),
    dcc.Interval(id='register-video-interval', interval=100, disabled=True, n_intervals=0),
    dcc.Interval(id='image-capture-interval', interval=500, disabled=True, n_intervals=0),
    dcc.Interval(id='attendance-video-interval', interval=150, disabled=True, n_intervals=0),
    dcc.Interval(id='recognition-processing-interval', interval=2000, disabled=True, n_intervals=0),
    html.Div(id="train-model-status-toast-area", style={"position": "fixed", "top": "70px", "right": "10px", "zIndex": 9999}),
    html.Div(id='page-content-wrapper', children=login_page)
])

# MODIFIED: Navbar updated to remove Dashboard and make Register the new default page.
def create_app_navbar():
    return dbc.Navbar(
        [
            dbc.NavbarBrand("Face Attendance System (S.N.J.B)", className="ms-2", href="/"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Register Student", href="/", active="exact")),
                dbc.NavItem(dbc.NavLink("Take Attendance", href="/attendance", active="exact")),
                dbc.NavItem(dbc.NavLink("Manage Students", href="/manage", active="exact")),
                dbc.NavItem(dbc.NavLink("Export Data", href="/export", active="exact")),
            ], pills=True, className="me-auto"),
            dbc.Button("Train Model (Manual)", id="train-model-button-nav", color="warning", className="me-2"),
            dbc.Button("Logout", id="logout-button", color="danger", className="me-2")
        ], color="dark", dark=True, className="mb-4"
    )

# REMOVED: The create_dashboard_page function is no longer needed.

def create_register_page():
    return html.Div([
        dbc.Row(dbc.Col(dbc.Card([
            dbc.CardHeader("Register New Student (for S.N.J.B Model)"),
            dbc.CardBody(dbc.Row([
                dbc.Col([
                    dbc.Input(id='student-name', placeholder="Full Name", className="mb-3"),
                    dbc.Select(id='student-branch', options=[{'label': s, 'value': v} for s, v in [
                        ('Computer Science', 'CS'), ('Electrical Engineering', 'EE'), ('Mechanical Engineering', 'ME'),
                        ('Civil Engineering', 'CE'), ('Business Administration', 'BA')]], placeholder="Select Branch", className="mb-3"),
                    dbc.Input(id='student-id', type="text", placeholder="Student ID / Roll Number (Numeric)", className="mb-3"),
                    dbc.Input(id='student-contact-lbph', type="text", placeholder="Contact (Optional for LBPH)", className="mb-3"),
                    dbc.ButtonGroup([
                        dbc.Button("Start Camera", id='start-cam-register', color="info", className="me-1"),
                        dbc.Button("Stop Camera", id='stop-cam-register', color="warning", disabled=True)
                    ], className="w-100 mb-2"),
                    dbc.Button("Start Capturing Images", id='capture-face-button-lbph', color="primary", className="w-100 mb-3", disabled=True),
                    html.Div(id='capture-status-register', className="text-center mb-3", children="Fill details and start camera."),
                    dbc.Progress(id="capture-progress", value=0, striped=True, animated=True, className="mb-3", style={'height': '20px'}),
                    dbc.Button("Register Student & Train Model", id='register-student-button-lbph', color="success", className="w-100", disabled=True),
                    html.Small(f"Captures {TARGET_IMAGE_COUNT} images. Ensure good lighting and clear face view.", className="d-block text-muted mt-2")
                ], md=6),
                dbc.Col([
                    html.Div(id='webcam-feed-container-register', className="text-center", children=[
                        html.Img(id='webcam-video-register',
                                 src=app.get_asset_url('webcam_placeholder.png') if os.path.exists('assets/webcam_placeholder.png') else "https://via.placeholder.com/320x240.png?text=Webcam+Off",
                                 style={'width': '320px', 'height': '240px', 'border': '1px solid grey'})]),
                    html.Div(id='captured-images-preview-area', className="text-center mt-3 d-flex flex-wrap justify-content-center")
                ], md=6)
            ]))])))
    ])

def create_attendance_page():
    return html.Div([
        dbc.Row(dbc.Col(dbc.Card([
            dbc.CardHeader("Take Attendance (S.N.J.B Recognition)"),
            dbc.CardBody(dbc.Row([
                dbc.Col([
                    html.Div(id='webcam-feed-container-attendance', className="text-center mb-3", children=[
                        html.Img(id='webcam-video-attendance',
                                 src=app.get_asset_url('webcam_placeholder.png') if os.path.exists('assets/webcam_placeholder.png') else "https://via.placeholder.com/320x240.png?text=Webcam+Off",
                                 style={'width': '320px', 'height': '240px', 'border': '1px solid grey'})]),
                    dbc.ButtonGroup([
                        dbc.Button("Start Camera & Recognition", id='start-recognition-attendance', color="primary", className="me-1"),
                        dbc.Button("Stop Camera & Recognition", id='stop-recognition-attendance', color="danger", disabled=True)
                    ], className="w-100 mb-3"),
                    html.Div(id='recognition-result-attendance', className="mt-4 p-3 border rounded text-center",
                             children="Camera and recognition are off. Ensure model is trained.")
                ], md=6),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Today's Attendance Records"),
                    dbc.CardBody(dash_table.DataTable(
                        id='attendance-records-table-live',
                        columns=[{'name': c, 'id': c} for c in ['student_id', 'name', 'branch', 'time', 'status']],
                        page_size=8, style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '8px', 'color': 'white'},
                        style_header={'backgroundColor': 'rgb(30,30,30)', 'color': 'white', 'fontWeight': 'bold'},
                        style_data={'backgroundColor': 'rgb(50,50,50)', 'color': 'white'}
                    ))]), md=6)
            ]))])))
    ])

def create_manage_page():
    return html.Div([
        dbc.Row(dbc.Col(dbc.Card([
            dbc.CardHeader("Manage Students (from App DB)"),
            dbc.CardBody(dbc.Row(dbc.Col([
                html.Div(id='manage-student-notification-area', className="mb-3"),
                dbc.Input(id='search-student', placeholder="Search by name or ID (Roll Number)", className="mb-3"),
                dash_table.DataTable(
                    id='students-table',
                    columns=[{'name': c, 'id': c} for c in ['student_id', 'name', 'branch', 'registration_date']],
                    row_selectable='single', selected_rows=[], page_size=10, style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '10px', 'color': 'white'},
                    style_header={'backgroundColor': 'rgb(30,30,30)', 'color': 'white', 'fontWeight': 'bold'},
                    style_data={'backgroundColor': 'rgb(50,50,50)', 'color': 'white'}
                ),
                dbc.Button("Delete Selected Student", id='delete-student-button', color="danger", className="mt-3", disabled=True),
                html.Small("Note: Deleting removes the student record and their images. Re-train the model afterwards for best results.", className="d-block text-muted mt-2")
            ])))])))
    ])

def create_export_page():
    return html.Div([
        dbc.Row(dbc.Col(dbc.Card([
            dbc.CardHeader("Export Attendance Data (from App DB)"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Date Range:"),
                        dcc.DatePickerRange(
                            id='date-range', min_date_allowed=datetime.now() - timedelta(days=365*2),
                            max_date_allowed=datetime.now(), start_date=(datetime.now() - timedelta(days=30)).date(),
                            end_date=datetime.now().date(), display_format='YYYY-MM-DD', className="mb-3 d-block"
                        )], md=6),
                    dbc.Col([
                        html.Label("Select Branch:"),
                        dbc.Select(id='export-branch', options=[{'label': 'All Branches', 'value': 'all'}] + [{'label': s, 'value': v} for s,v in [
                            ('Computer Science', 'CS'), ('Electrical Engineering', 'EE'), ('Mechanical Engineering', 'ME'),
                            ('Civil Engineering', 'CE'), ('Business Administration', 'BA')]], value='all', className="mb-3")
                    ], md=6)
                ]),
                dbc.Row(dbc.Col([
                    dbc.Button("Export to CSV", id='export-button', color="primary", className="mt-4 w-100"),
                    dcc.Download(id="download-csv")
                ])),
                dbc.Row(dbc.Col(html.Div(id='export-preview', className="mt-4")))
            ])])))
    ])

def frame_to_base64(frame, quality=90):
    if frame is None: return None
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')

# MODIFIED: Page router no longer creates a dashboard. Defaults to the register page.
@callback(
    Output('page-content-wrapper', 'children'),
    [Input('url', 'pathname')],
    [State('session', 'data'),
     State('register-video-interval', 'disabled'),
     State('attendance-video-interval', 'disabled')]
)
def display_page_router(pathname, session_data, reg_interval_disabled, att_interval_disabled):
    global CAPTURING_IN_PROGRESS
    if pathname not in ['/register', '/attendance', '/']: # Handle webcam release on navigation
        if not reg_interval_disabled or not att_interval_disabled:
            release_webcam()
    elif pathname != '/register' and pathname != '/':
        CAPTURING_IN_PROGRESS = False

    logged_in = session_data.get('logged_in', False)
    if not logged_in:
        return login_page

    navbar = create_app_navbar()
    page_specific_layout = None
    if pathname == '/' or pathname == '/register':
        page_specific_layout = create_register_page()
    elif pathname == '/attendance':
        page_specific_layout = create_attendance_page()
    elif pathname == '/manage':
        page_specific_layout = create_manage_page()
    elif pathname == '/export':
        page_specific_layout = create_export_page()
    elif pathname == '/login': # Redirect if already logged in
        page_specific_layout = create_register_page()
    else:
        page_specific_layout = dbc.Container(dbc.Alert("404: Page not found", color="danger"), fluid=True)
        
    return html.Div([navbar, dbc.Container(page_specific_layout, fluid=True, className="p-4")])

# MODIFIED: Login now redirects to '/' (the register page)
@callback(
    [Output('session', 'data', allow_duplicate=True),
     Output('url', 'pathname', allow_duplicate=True),
     Output('login-message', 'children')],
    [Input('login-button', 'n_clicks')],
    [State('username', 'value'), State('password', 'value'), State('session', 'data')],
    prevent_initial_call=True
)
def handle_login(n_clicks, username, password, session_data):
    if not n_clicks: return session_data, dash.no_update, ""
    conn = sqlite3.connect('attendance_app.db')
    c = conn.cursor()
    c.execute("SELECT * FROM admins WHERE username = ? AND password = ?", (username, password))
    admin = c.fetchone()
    conn.close()
    if admin:
        new_session_data = session_data.copy()
        new_session_data['logged_in'] = True
        new_session_data['username'] = username
        return new_session_data, '/', "" # Redirect to the new home page
    return session_data, dash.no_update, dbc.Alert("Invalid username or password", color="danger", dismissable=True)

@callback(
    [Output('session', 'data', allow_duplicate=True),
     Output('url', 'pathname', allow_duplicate=True)],
    [Input('logout-button', 'n_clicks')],
    [State('session', 'data')],
    prevent_initial_call=True
)
def handle_logout(n_clicks, session_data):
    if n_clicks:
        new_session_data = session_data.copy()
        new_session_data['logged_in'] = False
        new_session_data.pop('username', None)
        release_webcam()
        return new_session_data, '/login'
    return session_data, dash.no_update

# REMOVED: The main dashboard callback `update_dashboard_metrics` is no longer needed.

# --- MANAGE STUDENTS SECTION ---

@callback(
    Output('students-table', 'data'),
    [Input('url', 'pathname'),
     Input('search-student', 'value'),
     Input('delete-student-button', 'n_clicks')],
    [State('session', 'data')]
)
def update_students_table(pathname, search_term, delete_clicks, session_data):
    if not session_data.get('logged_in') or pathname != '/manage':
        return []

    conn = sqlite3.connect('attendance_app.db')
    try:
        query = "SELECT student_id, name, branch, registration_date FROM students ORDER BY name"
        params = []
        if search_term:
            query = "SELECT student_id, name, branch, registration_date FROM students WHERE name LIKE ? OR student_id LIKE ? ORDER BY name"
            params.extend([f'%{search_term}%', f'%{search_term}%'])

        students_df = pd.read_sql_query(query, conn, params=params if params else None)
        return students_df.to_dict('records')
    finally:
        conn.close()

@callback(
    Output('manage-student-notification-area', 'children'),
    Input('delete-student-button', 'n_clicks'),
    [State('students-table', 'selected_rows'),
     State('students-table', 'data')],
    prevent_initial_call=True
)
def handle_delete_student(n_clicks, selected_rows, table_data):
    if not n_clicks or not selected_rows:
        return dash.no_update

    selected_row_index = selected_rows[0]
    if selected_row_index >= len(table_data):
        return dbc.Alert("Selection is out of date. Please re-select the student to delete.", color="warning")

    student_to_delete = table_data[selected_row_index]
    student_id_to_delete = student_to_delete['student_id']
    student_name_to_delete = student_to_delete['name']

    conn = sqlite3.connect('attendance_app.db')
    try:
        c = conn.cursor()
        c.execute("DELETE FROM students WHERE student_id = ?", (student_id_to_delete,))
        c.execute("DELETE FROM attendance WHERE student_id = ?", (student_id_to_delete,))
        conn.commit()
        print(f"Deleted student {student_id_to_delete} from attendance_app.db")
    except Exception as e:
        conn.close()
        return dbc.Alert(f"Database error while deleting: {e}", color="danger")
    finally:
        conn.close()

    base_dir_my_fr = os.getcwd()
    is_mock_active = 'MockMyFr' in str(type(my_fr))
    if not is_mock_active and hasattr(my_fr, '__file__') and my_fr.__file__ is not None:
        base_dir_my_fr = os.path.dirname(os.path.abspath(my_fr.__file__))
    student_image_dir = os.path.join(base_dir_my_fr, my_fr.IMAGES_FOLDER)

    delete_count = 0
    for i in range(1, TARGET_IMAGE_COUNT + 10):
        img_path = os.path.join(student_image_dir, f"{student_id_to_delete}.{i}.jpg")
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
                delete_count += 1
            except Exception as e:
                print(f"Error deleting image {img_path}: {e}")
    print(f"Deleted {delete_count} images for student {student_id_to_delete}.")

    return dbc.Alert(f"Successfully deleted student '{student_name_to_delete}' (ID: {student_id_to_delete}) and {delete_count} images. Please retrain the model.", color="success", dismissable=True)

@callback(
    Output('delete-student-button', 'disabled'),
    Input('students-table', 'selected_rows'),
    prevent_initial_call=True
)
def update_delete_button_state(selected_rows):
    return not selected_rows

# --- OTHER CALLBACKS (Unchanged) ---

@callback(
    Output('train-model-status-toast-area', 'children', allow_duplicate=True),
    Input('train-model-button-nav', 'n_clicks'),
    prevent_initial_call=True
)
def handle_manual_train_model_global(n_clicks):
    if n_clicks:
        toast_message, toast_header, toast_icon = "", "", ""
        current_time_ms = int(time.time() * 1000)
        try:
            print("Manual Training model via Dash button...")
            training_success = my_fr.train_faces()

            base_dir_my_fr = os.getcwd()
            is_mock_active = 'MockMyFr' in str(type(my_fr))
            if not is_mock_active and hasattr(my_fr, '__file__') and my_fr.__file__ is not None:
                base_dir_my_fr = os.path.dirname(os.path.abspath(my_fr.__file__))
            model_file = os.path.join(base_dir_my_fr, my_fr.DATASET_FOLDER, 'trainer.yml')

            if training_success and os.path.exists(model_file):
                load_lbph_recognizer()
                toast_message, toast_header, toast_icon = "Model training completed successfully!", "Training Success", "success"
            elif training_success:
                toast_message, toast_header, toast_icon = "Training function ran, but trainer.yml not found. Check training script.", "Training Issue", "warning"
            else:
                toast_message, toast_header, toast_icon = "Model training reported failure (e.g., no images found in dataset).", "Training Failed", "danger"
        except Exception as e:
            print(f"Error during manual training: {e}")
            toast_message, toast_header, toast_icon = f"Error during model training: {str(e)}", "Training Error", "danger"
        return dbc.Toast(toast_message, id=f"manual-train-toast-{n_clicks}-{current_time_ms}", header=toast_header, icon=toast_icon, duration=5000, is_open=True)
    return dash.no_update

@callback(
    [Output('register-video-interval', 'disabled'),
     Output('start-cam-register', 'disabled'),
     Output('stop-cam-register', 'disabled'),
     Output('capture-face-button-lbph', 'disabled'),
     Output('webcam-video-register', 'src', allow_duplicate=True)],
    [Input('start-cam-register', 'n_clicks'),
     Input('stop-cam-register', 'n_clicks'),
     Input('url', 'pathname')],
    [State('register-video-interval', 'disabled'), State('session', 'data')],
    prevent_initial_call=True
)
def toggle_camera_register_cb(start_n, stop_n, pathname, interval_disabled, session_data):
    if not session_data.get('logged_in'): return True, True, True, True, dash.no_update
    ctx = dash.callback_context
    triggered_id = ctx.triggered_id.split('.')[0] if ctx.triggered_id else None
    ph_src = app.get_asset_url('webcam_placeholder.png') if os.path.exists('assets/webcam_placeholder.png') else "https://via.placeholder.com/320x240.png?text=Webcam+Off"

    if triggered_id == 'start-cam-register':
        get_webcam()
        return False, True, False, False, ph_src.replace("Off", "Starting...")
    elif triggered_id == 'stop-cam-register':
        release_webcam()
        return True, False, True, True, ph_src
    elif triggered_id == 'url' and pathname not in ['/', '/register'] and not interval_disabled:
        release_webcam()
        return True, False, True, True, ph_src
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

@callback(
    Output('webcam-video-register', 'src', allow_duplicate=True),
    Input('register-video-interval', 'n_intervals'),
    State('register-video-interval', 'disabled'),
    prevent_initial_call=True
)
def update_register_feed_cb(n_intervals, interval_disabled):
    if interval_disabled: return dash.no_update
    cam = get_webcam()
    if cam is None or not cam.isOpened():
        return app.get_asset_url('webcam_placeholder.png') if os.path.exists('assets/webcam_placeholder.png') else "https://via.placeholder.com/320x240.png?text=Webcam+Error"
    ret, frame = cam.read()
    if not ret or frame is None: return dash.no_update

    preview_frame = frame.copy()
    if my_fr.face_cascade:
        gray_frame_preview = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_preview = my_fr.face_cascade.detectMultiScale(gray_frame_preview, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        for (x, y, w, h) in faces_preview:
           cv2.rectangle(preview_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img_b64 = frame_to_base64(preview_frame)
    if img_b64:
        return f"data:image/jpeg;base64,{img_b64}"
    return dash.no_update

@callback(
    [Output('image-capture-interval', 'disabled'),
     Output('capture-status-register', 'children'),
     Output('capture-face-button-lbph', 'disabled', allow_duplicate=True),
     Output('register-student-button-lbph', 'disabled', allow_duplicate=True),
     Output('captured-image-count-store', 'data', allow_duplicate=True),
     Output('captured-images-preview-area', 'children'),
     Output('capture-progress', 'value'),
     Output('capture-progress', 'label')],
    [Input('capture-face-button-lbph', 'n_clicks')],
    [State('student-name', 'value'), State('student-branch', 'value'), State('student-id', 'value'),
     State('register-video-interval', 'disabled'),
     State('captured-image-count-store', 'data')],
    prevent_initial_call=True
)
def start_image_capture_process_cb(n_capture, name, branch, student_id_roll, cam_feed_disabled, current_capture_data):
    global CAPTURING_IN_PROGRESS
    if cam_feed_disabled:
        return True, dbc.Alert("Camera is off. Start camera first.", color="warning"), False, True, current_capture_data, [], 0, "0%"
    if not (name and branch and student_id_roll):
        return True, dbc.Alert("Please fill all student details first.", color="warning"), False, True, current_capture_data, [], 0, "0%"
    if not student_id_roll.isdigit():
        return True, dbc.Alert("Student ID / Roll Number must be numeric.", color="warning"), False, True, current_capture_data, [], 0, "0%"
    if not my_fr.face_cascade:
        return True, dbc.Alert("Face detection module (Haar cascade) not loaded. Cannot capture.", color="danger"), False, True, current_capture_data, [], 0, "0%"

    CAPTURING_IN_PROGRESS = True
    new_capture_data = {'student_id': student_id_roll, 'count': 0, 'filenames': []}

    base_dir = os.getcwd()
    is_mock_active = 'MockMyFr' in str(type(my_fr))
    if not is_mock_active and hasattr(my_fr, '__file__') and my_fr.__file__ is not None:
        base_dir = os.path.dirname(os.path.abspath(my_fr.__file__))
    student_image_dir = os.path.join(base_dir, my_fr.IMAGES_FOLDER)

    for i in range(1, TARGET_IMAGE_COUNT + 5):
        old_image_path = os.path.join(student_image_dir, f"{student_id_roll}.{i}.jpg")
        if os.path.exists(old_image_path):
            try:
                os.remove(old_image_path)
                print(f"Removed old image: {old_image_path}")
            except Exception as e:
                print(f"Error removing old image {old_image_path}: {e}")

    return False, dbc.Alert(f"Capturing images for {name} (ID: {student_id_roll})... Stay still.", color="info"), True, True, new_capture_data, [], 0, "0%"

@callback(
    [Output('image-capture-interval', 'disabled', allow_duplicate=True),
     Output('capture-status-register', 'children', allow_duplicate=True),
     Output('register-student-button-lbph', 'disabled', allow_duplicate=True),
     Output('captured-image-count-store', 'data', allow_duplicate=True),
     Output('captured-images-preview-area', 'children', allow_duplicate=True),
     Output('capture-progress', 'value', allow_duplicate=True),
     Output('capture-progress', 'label', allow_duplicate=True)],
    [Input('image-capture-interval', 'n_intervals')],
    [State('captured-image-count-store', 'data'),
     State('captured-images-preview-area', 'children')],
    prevent_initial_call=True
)
def handle_image_capture_interval_cb(n_intervals, capture_data, current_previews):
    global CAPTURING_IN_PROGRESS
    current_count = capture_data.get('count', 0) if isinstance(capture_data, dict) else 0
    progress_val = (current_count / TARGET_IMAGE_COUNT) * 100 if TARGET_IMAGE_COUNT > 0 else 0
    progress_label = f"{current_count}/{TARGET_IMAGE_COUNT}"

    if not CAPTURING_IN_PROGRESS or current_count >= TARGET_IMAGE_COUNT:
        CAPTURING_IN_PROGRESS = False
        msg = dbc.Alert(f"Finished capturing {current_count} images.", color="success") if current_count >= TARGET_IMAGE_COUNT else dash.no_update
        reg_btn_disabled = False if current_count >= TARGET_IMAGE_COUNT else True
        return True, msg, reg_btn_disabled, capture_data, current_previews, progress_val, progress_label

    cam = get_webcam()
    if cam is None or not cam.isOpened():
        CAPTURING_IN_PROGRESS = False
        return True, dbc.Alert("Webcam became unavailable during capture.", color="danger"), True, capture_data, current_previews, progress_val, progress_label
    if not my_fr.face_cascade:
        CAPTURING_IN_PROGRESS = False
        return True, dbc.Alert("Face detection module (Haar cascade) not loaded. Capture aborted.", color="danger"), True, capture_data, current_previews, progress_val, progress_label

    ret, frame = cam.read()
    if not ret or frame is None:
        return False, dash.no_update, True, capture_data, current_previews, progress_val, progress_label

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = my_fr.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=8, minSize=(80,80))
    status_msg = dash.no_update

    if isinstance(faces, np.ndarray) and faces.size > 0 and len(faces) == 1:
        (x, y, w, h) = faces[0]
        face_roi_gray_np = gray_frame[y:y + h, x:x + w]

        if face_roi_gray_np.size > 0:
            capture_data['count'] += 1
            student_id = capture_data['student_id']
            image_filename = f"{student_id}.{capture_data['count']}.jpg"

            base_dir = os.getcwd()
            is_mock_active = 'MockMyFr' in str(type(my_fr))
            if not is_mock_active and hasattr(my_fr, '__file__') and my_fr.__file__ is not None :
                base_dir = os.path.dirname(os.path.abspath(my_fr.__file__))
            filepath = os.path.join(base_dir, my_fr.IMAGES_FOLDER, image_filename)

            try:
                cv2.imwrite(filepath, face_roi_gray_np)
                if 'filenames' not in capture_data or not isinstance(capture_data['filenames'], list):
                    capture_data['filenames'] = []
                capture_data['filenames'].append(image_filename)
                print(f"Saved image: {filepath}")

                roi_b64_preview = frame_to_base64(cv2.resize(face_roi_gray_np, (60,60)), quality=70)
                if roi_b64_preview:
                    new_thumbnail = html.Img(src=f"data:image/jpeg;base64,{roi_b64_preview}", style={'width':'60px','height':'auto','margin':'2px','border':'1px solid #3498db'})
                    if not isinstance(current_previews, list): current_previews = []
                    current_previews.append(new_thumbnail)

                status_msg = dbc.Alert(f"Captured image {capture_data['count']}/{TARGET_IMAGE_COUNT} for ID {student_id}.", color="info")
                progress_val = (capture_data['count'] / TARGET_IMAGE_COUNT) * 100 if TARGET_IMAGE_COUNT > 0 else 0
                progress_label = f"{capture_data['count']}/{TARGET_IMAGE_COUNT}"

                if capture_data['count'] >= TARGET_IMAGE_COUNT:
                    CAPTURING_IN_PROGRESS = False
                    return True, dbc.Alert(f"All {TARGET_IMAGE_COUNT} images captured! Ready to Register.", color="success"), False, capture_data, current_previews, progress_val, progress_label
                return False, status_msg, True, capture_data, current_previews, progress_val, progress_label
            except Exception as e:
                print(f"Error saving image {filepath}: {e}")
                status_msg = dbc.Alert(f"Error saving image {capture_data['count']}. Retrying.", color="warning")
                capture_data['count'] -= 1
                return False, status_msg, True, capture_data, current_previews, progress_val, progress_label
        else:
            status_msg = dbc.Alert("Face ROI was empty. Adjust view.", color="warning")
    elif isinstance(faces, np.ndarray) and len(faces) > 1:
        status_msg = dbc.Alert("Multiple faces detected. Please ensure only one person.", color="warning")
    else:
        status_msg = dbc.Alert("No face detected. Adjust view.", color="warning")
    return False, status_msg, True, capture_data, current_previews, progress_val, progress_label

@callback(
    [Output('capture-status-register', 'children', allow_duplicate=True),
     Output('student-name', 'value', allow_duplicate=True),
     Output('student-branch', 'value', allow_duplicate=True),
     Output('student-id', 'value', allow_duplicate=True),
     Output('student-contact-lbph', 'value', allow_duplicate=True),
     Output('captured-image-count-store', 'data', allow_duplicate=True),
     Output('captured-images-preview-area', 'children', allow_duplicate=True),
     Output('register-student-button-lbph', 'disabled', allow_duplicate=True),
     Output('capture-face-button-lbph', 'disabled', allow_duplicate=True),
     Output('train-model-status-toast-area', 'children', allow_duplicate=True),
     Output('capture-progress', 'value', allow_duplicate=True),
     Output('capture-progress', 'label', allow_duplicate=True)],
    [Input('register-student-button-lbph', 'n_clicks')],
    [State('student-name', 'value'), State('student-branch', 'value'), State('student-id', 'value'),
     State('student-contact-lbph', 'value'), State('captured-image-count-store', 'data')],
    prevent_initial_call=True
)
def register_db_and_train_cb(n_clicks, name, branch, student_id_roll, contact, capture_data):
    no_up = dash.no_update
    reset_capture_store = {'student_id': None, 'count': 0, 'filenames': []}
    toast_output = no_up
    current_time_ms = int(time.time() * 1000)

    current_progress_val = (capture_data.get('count', 0) / TARGET_IMAGE_COUNT) * 100 if isinstance(capture_data, dict) and TARGET_IMAGE_COUNT > 0 else 0
    current_progress_label = f"{capture_data.get('count', 0)}/{TARGET_IMAGE_COUNT}" if isinstance(capture_data, dict) else "0/10"

    if not (name and branch and student_id_roll):
        return dbc.Alert("Fill all required fields.", color="danger"), name, branch, student_id_roll, contact, capture_data, no_up, False, False, toast_output, current_progress_val, current_progress_label
    if not student_id_roll.isdigit():
        return dbc.Alert("Student ID must be numeric.", color="danger"), name, branch, student_id_roll, contact, capture_data, no_up, False, False, toast_output, current_progress_val, current_progress_label
    if not isinstance(capture_data, dict) or not capture_data.get('student_id') or capture_data['student_id'] != student_id_roll or capture_data.get('count', 0) < TARGET_IMAGE_COUNT:
        return dbc.Alert(f"Capture {TARGET_IMAGE_COUNT} images for Student ID {student_id_roll} first.", color="danger"), name, branch, student_id_roll, contact, capture_data, no_up, True, False, toast_output, current_progress_val, current_progress_label

    try:
        my_fr.database.add_student(name, contact if contact else "", student_id_roll, capture_data['filenames'])
        print(f"Student {name} ({student_id_roll}) added to my_fr.database with {len(capture_data['filenames'])} images.")

        conn_dash_db = sqlite3.connect('attendance_app.db')
        c_dash = conn_dash_db.cursor()
        reg_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            c_dash.execute("INSERT INTO students (name, branch, student_id, registration_date) VALUES (?, ?, ?, ?)",
                           (name, branch, student_id_roll, reg_date))
            conn_dash_db.commit()
            print(f"Student {name} (ID: {student_id_roll}) INSERTED into attendance_app.db")
        except sqlite3.IntegrityError:
            conn_dash_db.rollback()
            print(f"Student ID {student_id_roll} already in Dash App DB or other SQLite error.")
            return dbc.Alert(f"Student ID {student_id_roll} already exists in the system.", color="warning"), name, branch, student_id_roll, contact, reset_capture_store, [], True, False, toast_output, 0, "0%"
        finally:
            conn_dash_db.close()

        try:
            print(f"Auto-training model after registering {name}...")
            training_success = my_fr.train_faces()

            base_dir_my_fr = os.getcwd()
            is_mock_active = 'MockMyFr' in str(type(my_fr))
            if not is_mock_active and hasattr(my_fr, '__file__') and my_fr.__file__ is not None:
                base_dir_my_fr = os.path.dirname(os.path.abspath(my_fr.__file__))
            model_file = os.path.join(base_dir_my_fr, my_fr.DATASET_FOLDER, 'trainer.yml')

            if training_success and os.path.exists(model_file):
                load_lbph_recognizer()
                train_msg, train_hdr, train_icon = "Model auto-trained successfully!", "Auto-Training Success", "success"
            elif training_success:
                train_msg, train_hdr, train_icon = "Auto-Training function ran, but trainer.yml not found.", "Auto-Training Issue", "warning"
            else:
                train_msg, train_hdr, train_icon = "Auto-Training reported failure (e.g., no images found).", "Auto-Training Failed", "danger"
            toast_output = dbc.Toast(train_msg, id=f"auto-train-toast-{n_clicks}-{current_time_ms}", header=train_hdr, icon=train_icon, duration=5000, is_open=True)
        except Exception as e_train:
            print(f"Error during auto-training: {e_train}")
            toast_output = dbc.Toast(f"Error during auto-training: {str(e_train)}", id=f"auto-train-toast-err-{n_clicks}-{current_time_ms}", header="Training Error", icon="danger", duration=5000, is_open=True)

        return dbc.Alert(f"Student {name} registered. Model training attempted.", color="success"), "", None, "", "", reset_capture_store, [], True, False, toast_output, 0, "0%"
    except Exception as e_reg:
        print(f"Overall Registration Error: {e_reg}")
        return dbc.Alert(f"Registration Error: {str(e_reg)}", color="danger"), name, branch, student_id_roll, contact, capture_data, no_up, False, False, toast_output, current_progress_val, current_progress_label

@callback(
    [Output('attendance-video-interval', 'disabled'),
     Output('recognition-processing-interval', 'disabled'),
     Output('start-recognition-attendance', 'disabled'),
     Output('stop-recognition-attendance', 'disabled'),
     Output('webcam-video-attendance', 'src', allow_duplicate=True),
     Output('recognition-result-attendance', 'children', allow_duplicate=True),
     Output('attendance-records-table-live', 'data', allow_duplicate=True)],
    [Input('start-recognition-attendance', 'n_clicks'),
     Input('stop-recognition-attendance', 'n_clicks'),
     Input('url', 'pathname')],
    [State('attendance-video-interval', 'disabled'), State('session', 'data')],
    prevent_initial_call=True
)
def toggle_recognition_attendance_cb(start_n, stop_n, pathname, video_interval_disabled, session_data):
    if not session_data.get('logged_in'): return True, True, True, True, dash.no_update, dash.no_update, dash.no_update
    ctx = dash.callback_context
    triggered_id = ctx.triggered_id.split('.')[0] if ctx.triggered_id else None
    placeholder_src = app.get_asset_url('webcam_placeholder.png') if os.path.exists('assets/webcam_placeholder.png') else "https://via.placeholder.com/320x240.png?text=Webcam+Off"
    if triggered_id == 'start-recognition-attendance':
        if not load_lbph_recognizer():
             return True, True, False, True, placeholder_src, dbc.Alert("Failed to load model. Train first.", color="danger"), []
        if not my_fr.face_cascade:
            return True, True, False, True, placeholder_src, dbc.Alert("Face detection module (Haar) not loaded. Attendance will not work.", color="danger"), []

        get_webcam()
        conn_dash_db = sqlite3.connect('attendance_app.db')
        today_str = datetime.now().strftime("%Y-%m-%d")
        df = pd.read_sql_query(f"SELECT s.student_id, s.name, s.branch, a.time, a.status FROM attendance a JOIN students s ON a.student_id = s.student_id WHERE a.date = '{today_str}' ORDER BY a.time DESC", conn_dash_db)
        conn_dash_db.close()
        return False, False, True, False, placeholder_src.replace("Off", "Starting..."), dbc.Alert("Recognition active...", color="info"), df.to_dict('records')
    elif triggered_id == 'stop-recognition-attendance':
        release_webcam()
        return True, True, False, True, placeholder_src, dbc.Alert("Cam/Recog stopped.", color="warning"), dash.no_update
    elif triggered_id == 'url' and pathname != '/attendance' and not video_interval_disabled:
        release_webcam()
        return True, True, False, True, placeholder_src, dbc.Alert("Cam stopped due to navigation.", color="secondary"), dash.no_update
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

@callback(
    [Output('webcam-video-attendance', 'src', allow_duplicate=True),
     Output('recognition-result-attendance', 'children', allow_duplicate=True),
     Output('attendance-records-table-live', 'data', allow_duplicate=True)],
    [Input('attendance-video-interval', 'n_intervals'),
     Input('recognition-processing-interval', 'n_intervals')],
    [State('attendance-video-interval', 'disabled'), State('recognition-processing-interval', 'disabled')],
    prevent_initial_call=True
)
def process_attendance_feed_and_recognition_cb(video_tick, recognition_tick, video_disabled, recognition_disabled):
    ctx = dash.callback_context
    triggered_id = ctx.triggered_id.split('.')[0] if ctx.triggered_id else None
    new_feed_src, new_recognition_result, new_table_data = dash.no_update, dash.no_update, dash.no_update
    placeholder_img = app.get_asset_url('webcam_placeholder.png') if os.path.exists('assets/webcam_placeholder.png') else "https://via.placeholder.com/320x240.png?text=Webcam+Off"

    cam = get_webcam()
    if cam is None or not cam.isOpened() or video_disabled:
        if triggered_id == 'attendance-video-interval' and not video_disabled:
             return placeholder_img, dbc.Alert("Camera is off or has become unavailable.", color="secondary"), []
        return dash.no_update, dash.no_update, dash.no_update

    ret, frame = cam.read()
    if not ret or frame is None:
        return dash.no_update, dbc.Alert("Error reading frame.", color="danger"), dash.no_update

    if not my_fr.face_cascade:
        current_feed_b64 = frame_to_base64(frame) if frame is not None else placeholder_img
        err_msg = dbc.Alert("Face detection module (Haar) not loaded. Recognition will fail.", color="danger")
        if triggered_id == 'attendance-video-interval' and not video_disabled:
            return f"data:image/jpeg;base64,{current_feed_b64}", err_msg, dash.no_update
        return dash.no_update, err_msg, dash.no_update

    processed_frame = frame.copy()
    gray_frame_att = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_att = my_fr.face_cascade.detectMultiScale(gray_frame_att, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    recognition_messages = []

    if triggered_id == 'recognition-processing-interval' and not recognition_disabled:
        if RECOGNIZER_LBPH is None:
            new_recognition_result = dbc.Alert("LBPH Model not loaded/available. Train model first.", color="danger")
        elif RECOGNIZER_LBPH == "MockRecognizerLoaded":
            new_recognition_result = dbc.Alert("Mock Recognizer active. Real recognition disabled.", color="info")
            if isinstance(faces_att, np.ndarray) and faces_att.size > 0:
                (x, y, w, h) = faces_att[0]
                cv2.rectangle(processed_frame, (x,y), (x+w,y+h), (0,255,255), 2)
                cv2.putText(processed_frame, "Mock: StudentX (C:50)", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                recognition_messages.append("Mock recognized StudentX")
            else: recognition_messages.append("Mock: No faces to mock-recognize.")
            if recognition_messages: new_recognition_result = dbc.Alert(", ".join(recognition_messages), color="info", duration=3000)
            conn_dash_db_mock = sqlite3.connect('attendance_app.db')
            try:
                df_table_mock = pd.read_sql_query(f"SELECT s.student_id, s.name, s.branch, a.time, a.status FROM attendance a JOIN students s ON a.student_id = s.student_id WHERE a.date = '{datetime.now().strftime('%Y-%m-%d')}' ORDER BY a.time DESC", conn_dash_db_mock)
                new_table_data = df_table_mock.to_dict('records')
            finally: conn_dash_db_mock.close()

        elif isinstance(faces_att, np.ndarray) and faces_att.size > 0:
            found_match_in_frame = False
            conn_dash_db = sqlite3.connect('attendance_app.db')
            try:
                for (x, y, w, h) in faces_att:
                    if hasattr(RECOGNIZER_LBPH, 'predict'):
                        face_id_pred, confidence = RECOGNIZER_LBPH.predict(gray_frame_att[y:y + h, x:x + w])
                        print(f"Real Recognition - ID: {face_id_pred}, Confidence: {confidence}")
                        name_display = "Unknown"
                        if confidence < 65:
                            c_dash = conn_dash_db.cursor()
                            c_dash.execute("SELECT name, branch FROM students WHERE student_id = ?", (str(face_id_pred),))
                            student_info_dash = c_dash.fetchone()
                            if student_info_dash:
                                name_display = student_info_dash[0]
                                student_id_roll = str(face_id_pred)
                                cur_date, cur_time = datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M:%S")
                                c_dash.execute("SELECT id FROM attendance WHERE student_id = ? AND date = ?", (student_id_roll, cur_date))
                                if not c_dash.fetchone():
                                    c_dash.execute("INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)", (student_id_roll, cur_date, cur_time, 'Present'))
                                    conn_dash_db.commit()
                                    recognition_messages.append(f"{name_display} ({student_id_roll}) Present (C:{confidence:.0f})")
                                    found_match_in_frame = True
                                else: recognition_messages.append(f"{name_display} ({student_id_roll}) already marked (C:{confidence:.0f})")
                            else: recognition_messages.append(f"ID {face_id_pred} (C:{confidence:.0f}), not in App DB."); name_display = f"ID:{face_id_pred} (Not in DB)"
                        else: recognition_messages.append(f"Low confidence for ID:{face_id_pred} (C:{confidence:.0f})")
                        cv2.rectangle(processed_frame, (x,y), (x+w,y+h), (0,255,0) if name_display != "Unknown" and "Not in DB" not in name_display else (0,0,255), 2)
                        cv2.putText(processed_frame, f"{name_display} C:{confidence:.0f}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                    else:
                        recognition_messages.append("Recognizer object invalid.")
                        break
                if recognition_messages: new_recognition_result = dbc.Alert(", ".join(recognition_messages), color="success" if found_match_in_frame else "info", duration=3000)
                else: new_recognition_result = dbc.Alert("Faces detected, but no confident recognitions.", color="secondary")

                df_table = pd.read_sql_query(f"SELECT s.student_id, s.name, s.branch, a.time, a.status FROM attendance a JOIN students s ON a.student_id = s.student_id WHERE a.date = '{datetime.now().strftime('%Y-%m-%d')}' ORDER BY a.time DESC", conn_dash_db)
                new_table_data = df_table.to_dict('records')
            finally: conn_dash_db.close()
        else:
            if not recognition_disabled: new_recognition_result = dbc.Alert("No faces in view for recognition.", color="light")

    elif isinstance(faces_att, np.ndarray) and faces_att.size > 0 and not video_disabled :
         for (x, y, w, h) in faces_att: cv2.rectangle(processed_frame, (x,y), (x+w,y+h), (255,0,0),2)

    if not video_disabled:
        img_b64_processed = frame_to_base64(processed_frame)
        new_feed_src = f"data:image/jpeg;base64,{img_b64_processed}" if img_b64_processed else placeholder_img

    return new_feed_src, new_recognition_result, new_table_data

@callback(
    [Output('export-preview', 'children'), Output('download-csv', 'data')],
    [Input('export-button', 'n_clicks')],
    [State('date-range', 'start_date'), State('date-range', 'end_date'), 
     State('export-branch', 'value'), State('session', 'data')],
    prevent_initial_call=True
)
def export_data_csv_global(n_clicks, start_date, end_date, branch, session_data):
    if not session_data.get('logged_in') or not n_clicks:
        return dash.no_update, dash.no_update

    # --- Step 1: Query the database (same as before) ---
    conn = sqlite3.connect('attendance_app.db')
    try:
        query = "SELECT a.date, a.time, s.student_id, s.name, s.branch, a.status FROM attendance a JOIN students s ON a.student_id = s.student_id WHERE date(a.date) BETWEEN date(?) AND date(?)"
        params = [start_date, end_date]
        if branch != 'all':
            query += " AND s.branch = ?"
            params.append(branch)
        query += " ORDER BY a.date DESC, a.time DESC"
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        return dbc.Alert("No data for selected criteria. Nothing to export.", color="warning"), dash.no_update

    # --- Step 2: Define file path and save to the current server folder ---
    filename = f"attendance_{start_date}_{end_date}_{branch}.csv"
    
    # Get the current working directory (where your script is running)
    current_folder = os.getcwd()
    filepath = os.path.join(current_folder, filename)
    
    user_notification = None
    try:
        # Save the DataFrame to a CSV file in the current folder
        df.to_csv(filepath, index=False)
        print(f"Export successful. File saved at: {filepath}")
        
        # Create a success message for the UI
        user_notification = dbc.Alert(
            f"Success! File saved on the server in the current folder: {filename}",
            color="success",
            dismissable=True
        )
    except Exception as e:
        print(f"Error saving file to server: {e}")
        # Create an error message for the UI if saving fails
        user_notification = dbc.Alert(
            f"Error saving file to the server: {e}",
            color="danger",
            dismissable=True
        )

    # --- Step 3: Send the data to the user's browser for download (same as before) ---
    download_data = dcc.send_data_frame(df.to_csv, filename, index=False)
    
    # Return the UI notification and the download data
    return user_notification, download_data

if __name__ == '__main__':
    assets_dir = 'assets'
    if not os.path.exists(assets_dir): os.makedirs(assets_dir)
    for img_name, size, color in [('webcam_placeholder.png',(320,240),'lightgrey'), ('login_icon.png',(150,150),'darkgrey')]:
        path = os.path.join(assets_dir, img_name)
        if not os.path.exists(path):
            try: Image.new('RGB', size, color=color).save(path)
            except Exception as e: print(f"Could not create {img_name}: {e}")

    is_mock_active_main = 'MockMyFr' in str(type(my_fr))
    if not is_mock_active_main:
        if hasattr(my_fr, '__file__') and my_fr.__file__ is not None and hasattr(my_fr,'IMAGES_FOLDER') and hasattr(my_fr,'DATASET_FOLDER'):
            base_dir_my_fr = os.path.dirname(os.path.abspath(my_fr.__file__))
            os.makedirs(os.path.join(base_dir_my_fr, my_fr.IMAGES_FOLDER), exist_ok=True)
            os.makedirs(os.path.join(base_dir_my_fr, my_fr.DATASET_FOLDER), exist_ok=True)
            print(f"Checked/created REAL LBPH folders relative to: {base_dir_my_fr}")
        elif hasattr(my_fr,'IMAGES_FOLDER') and hasattr(my_fr,'DATASET_FOLDER'):
            os.makedirs(my_fr.IMAGES_FOLDER, exist_ok=True); os.makedirs(my_fr.DATASET_FOLDER, exist_ok=True)
            print(f"Checked/created REAL LBPH folders based on my_fr paths (current dir context).")
        else:
             print("Warning: Real my_fr module does not have necessary attributes for folder creation. Check your my_fr.py")

    load_lbph_recognizer()
    import atexit
    atexit.register(release_webcam)
    app.run(debug=True,dev_tools_ui=False)
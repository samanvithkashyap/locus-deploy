import asyncio
import sys

# --- WINDOWS EVENT LOOP FIX ---
try:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass

import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import pickle
import av
import time
from collections import deque
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import gspread
from google.oauth2.service_account import Credentials

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_FILE = os.path.join(BASE_DIR, "encodings_mobilenet.pickle")
COSINE_THRESHOLD = 0.50 
FRAME_SKIP = 2
DETECT_FRAME_SKIP = 2
PROC_WIDTH = 640

# --- GOOGLE SHEETS SETUP ---
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

@st.cache_resource
def get_google_sheets_client():
    """Initialize Google Sheets client from Streamlit secrets"""
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=SCOPES
        )
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"Failed to connect to Google Sheets: {e}")
        return None

def get_or_create_attendance_sheet(client, sheet_name=None):
    """Get or create today's attendance sheet"""
    if client is None:
        return None
    
    if sheet_name is None:
        sheet_name = st.secrets.get("sheet_name", "LOCUS_Attendance")
    
    try:
        # Try to open existing spreadsheet BY ID (better than name)
        sheet_id = st.secrets.get("sheet_id", None)
        if sheet_id:
            spreadsheet = client.open_by_key(sheet_id)
        else:
            spreadsheet = client.open(sheet_name)
    except gspread.SpreadsheetNotFound:
        st.error(f"❌ Spreadsheet '{sheet_name}' not found. Please create it manually and add the sheet_id to secrets!")
        return None
    except Exception as e:
        st.error(f"❌ Error opening spreadsheet: {e}")
        return None
    
    # Get or create today's worksheet
    today = datetime.now().strftime('%Y-%m-%d')
    
    try:
        worksheet = spreadsheet.worksheet(today)
    except gspread.WorksheetNotFound:
        try:
            worksheet = spreadsheet.add_worksheet(title=today, rows=100, cols=4)
            # Add headers
            worksheet.update('A1:D1', [['Name', 'USN', 'Time', 'Status']])
        except Exception as e:
            st.error(f"❌ Could not create worksheet: {e}. Using first sheet.")
            worksheet = spreadsheet.get_worksheet(0)
    
    return worksheet

def mark_attendance_sheets(folder_name, worksheet):
    """Mark attendance in Google Sheets"""
    if worksheet is None:
        return False, "Error: Not connected to sheets"
    
    name, usn = get_student_details(folder_name)
    
    try:
        # Get all existing records
        records = worksheet.get_all_records()
        
        # Check if already marked
        for record in records:
            if record.get('Name') == name:
                return False, name
        
        # Add new attendance record
        now = datetime.now().strftime('%H:%M:%S')
        worksheet.append_row([name, usn, now, 'Present'])
        return True, name
        
    except Exception as e:
        st.error(f"Error marking attendance: {e}")
        return False, name

def get_present_students(worksheet):
    """Get list of present students from Google Sheets"""
    if worksheet is None:
        return []
    
    try:
        records = worksheet.get_all_records()
        return [record['Name'] for record in records if record.get('Name')]
    except:
        return []

# --- HELPER FUNCTIONS ---
def get_student_details(folder_name):
    clean_name = folder_name.strip()
    if "_" in clean_name:
        parts = clean_name.split("_", 1)
        return parts[1].replace("_", " "), parts[0]
    return clean_name, "N/A"

# --- LOAD DATABASE ---
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        db = pickle.load(f)
    all_student_names = set()
    for s in db["names"]:
        n, _ = get_student_details(s)
        all_student_names.add(n)
else:
    db = {"features": [], "names": []}
    all_student_names = set()

# Initialize Google Sheets
sheets_client = get_google_sheets_client()
attendance_sheet = get_or_create_attendance_sheet(sheets_client)

# --- AI PROCESSOR ---
class BlinkProcessor:
    def __init__(self):
        self.frame_count = 0
        self.is_verified_live = False
        self.status_msg = "Look at Camera"
        self.status_color = (255, 255, 255)
        self.model_loaded = False
        
        # Liveness detection state
        self.prev_face_gray = None
        self.motion_frames = deque(maxlen=8)
        self.center_x_history = deque(maxlen=10)
        self.liveness_check_count = 0
        
        # Face tracking
        self.last_face = None
        self.last_face_frame = 0
        self.max_face_age = 5
        self.last_recog_frame = 0
        self.face_lost_frames = 0
        
        # Load models
        try:
            detector_path = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
            self.detector = cv2.FaceDetectorYN.create(
                detector_path, "", (320, 320), 0.9, 0.3, 5000
            )
        except Exception as e:
            self.detector = None
            self.status_msg = f"ERROR: Detector Missing"

        try:
            recognizer_path = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
            self.recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
        except Exception as e:
            self.recognizer = None
            if self.status_msg == "Look at Camera":
                self.status_msg = f"ERROR: Recognizer Missing"

        self.model_loaded = self.detector is not None and self.recognizer is not None

        # Counter Logic
        self.total_students = len(all_student_names)
        self.present_count = 0
        self.present_set = set()
        
        # Load present students from Google Sheets
        if attendance_sheet:
            present_list = get_present_students(attendance_sheet)
            valid_present = [x for x in present_list if x in all_student_names]
            self.present_set = set(valid_present)
            self.present_count = len(self.present_set)

    def reset_liveness(self):
        """Reset liveness detection state"""
        self.prev_face_gray = None
        self.motion_frames.clear()
        self.center_x_history.clear()
        self.liveness_check_count = 0
        self.is_verified_live = False

    def check_liveness(self, img, face_box):
        """Simplified liveness check - motion + head movement"""
        try:
            x, y, w, h = face_box.astype(int)
            
            x = max(0, x)
            y = max(0, y)
            w = min(img.shape[1] - x, w)
            h = min(img.shape[0] - y, h)
            
            if w <= 0 or h <= 0:
                return False

            face_roi = img[y:y+h, x:x+w]
            if face_roi.size == 0:
                return False

            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            if self.prev_face_gray is not None and self.prev_face_gray.shape == gray.shape:
                diff = cv2.absdiff(gray, self.prev_face_gray)
                motion_score = float(np.mean(diff))
                self.motion_frames.append(motion_score)
            
            self.prev_face_gray = gray.copy()

            center_x = x + (w / 2.0)
            self.center_x_history.append(center_x)
            
            self.liveness_check_count += 1

            if self.liveness_check_count >= 4 and len(self.motion_frames) >= 4:
                avg_motion = sum(self.motion_frames) / len(self.motion_frames)
                
                if len(self.center_x_history) >= 2:
                    center_span = max(self.center_x_history) - min(self.center_x_history)
                    
                    if avg_motion > 2.5 and center_span > (w * 0.08):
                        return True
            
            if self.liveness_check_count >= 15 and len(self.motion_frames) >= 8:
                avg_motion = sum(self.motion_frames) / len(self.motion_frames)
                if avg_motion > 1.5:
                    return True

            return False
            
        except Exception as e:
            print(f"Liveness check error: {e}")
            return False

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            img = cv2.flip(img, 1)

            if not self.model_loaded:
                cv2.putText(img, self.status_msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            should_process = (self.frame_count % FRAME_SKIP == 0)
            
            h, w, _ = img.shape
            face = None

            proc_img = img
            scale_x = 1.0
            scale_y = 1.0
            if w > PROC_WIDTH:
                scale_x = w / float(PROC_WIDTH)
                proc_h = int(h / scale_x)
                proc_img = cv2.resize(img, (PROC_WIDTH, proc_h))
                scale_y = h / float(proc_h)

            def scale_face(face_arr, sx, sy):
                if face_arr is None:
                    return None
                scaled = face_arr.copy()
                scaled[0] *= sx
                scaled[1] *= sy
                scaled[2] *= sx
                scaled[3] *= sy
                for i in range(5, len(scaled), 2):
                    scaled[i] *= sx
                    if i + 1 < len(scaled):
                        scaled[i + 1] *= sy
                return scaled

            if should_process:
                should_detect = (self.frame_count % (FRAME_SKIP * DETECT_FRAME_SKIP) == 0) or self.last_face is None
                
                if should_detect:
                    self.detector.setInputSize((proc_img.shape[1], proc_img.shape[0]))
                    faces = self.detector.detect(proc_img)
                    
                    if faces[1] is not None and len(faces[1]) > 0:
                        face = scale_face(faces[1][0], scale_x, scale_y)
                        self.last_face = face
                        self.last_face_frame = self.frame_count
                        self.face_lost_frames = 0
                    else:
                        self.face_lost_frames += 1
                        if self.face_lost_frames > 3:
                            self.last_face = None
                            self.reset_liveness()
                else:
                    if self.last_face is not None and (self.frame_count - self.last_face_frame) <= self.max_face_age:
                        face = self.last_face
                    else:
                        self.last_face = None
                        self.reset_liveness()

            if face is None and self.last_face is not None:
                if (self.frame_count - self.last_face_frame) <= self.max_face_age:
                    face = self.last_face

            if face is not None:
                box = face[0:4]
                
                # PHASE 1: LIVENESS CHECK
                if not self.is_verified_live:
                    if should_process:
                        if self.check_liveness(img, box):
                            self.is_verified_live = True
                            self.status_msg = "LIVE! IDENTIFYING..."
                            self.status_color = (0, 255, 0)
                        else:
                            self.status_msg = "TURN HEAD SLOWLY"
                            self.status_color = (0, 165, 255)
                
                # PHASE 2: RECOGNITION
                elif should_process and (self.frame_count - self.last_recog_frame) >= (FRAME_SKIP * 2):
                    try:
                        aligned = self.recognizer.alignCrop(img, face)
                        feat = self.recognizer.feature(aligned)
                        self.last_recog_frame = self.frame_count
                        
                        best_score = 0.0
                        best_name = "Unknown"
                        
                        for idx, db_feat in enumerate(db["features"]):
                            score = self.recognizer.match(feat, db_feat, cv2.FaceRecognizerSF_FR_COSINE)
                            if score > best_score:
                                best_score = score
                                best_name = db["names"][idx]
                        
                        if best_score > COSINE_THRESHOLD:
                            marked, final_name = mark_attendance_sheets(best_name, attendance_sheet)
                            if final_name not in self.present_set:
                                self.present_set.add(final_name)
                                self.present_count += 1
                            
                            if marked:
                                self.status_msg = f"MARKED: {final_name}"
                                self.status_color = (0, 255, 0)
                            else:
                                self.status_msg = f"ALREADY MARKED: {final_name}"
                                self.status_color = (255, 255, 0)
                        else:
                            self.status_msg = "UNKNOWN IDENTITY"
                            self.status_color = (0, 0, 255)
                    except Exception as e:
                        print(f"Recognition error: {e}")
                        self.status_msg = "PROCESSING ERROR"
                        self.status_color = (255, 0, 0)
            else:
                self.status_msg = "NO FACE DETECTED"
                self.status_color = (200, 200, 200)

            # Draw Status Bar
            bar_h = 60
            cv2.rectangle(img, (0, h - bar_h), (w, h), (0, 0, 0), -1)
            cv2.putText(img, self.status_msg, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.status_color, 2)
            
            remaining = self.total_students - self.present_count
            counter_text = f"Pending: {max(0, remaining)}"
            (tw, th), _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(img, counter_text, (w - tw - 20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame

# --- UI SETUP ---
st.set_page_config(page_title="LOCUS", layout="wide")
st.title("LOCUS")

# Show Google Sheets status
if sheets_client and attendance_sheet:
    st.success("✅ Connected to Google Sheets")
    sheet_url = f"https://docs.google.com/spreadsheets/d/{attendance_sheet.spreadsheet.id}"
    st.markdown(f"[📊 View Attendance Sheet]({sheet_url})")
else:
    st.error("❌ Not connected to Google Sheets - Check secrets configuration")

st.sidebar.header("Absentees List")
if st.sidebar.button("🔄 Sync List"):
    st.rerun()

all_students_clean = sorted(list(all_student_names))
present_students = get_present_students(attendance_sheet) if attendance_sheet else []

absentees = sorted(list(set(all_students_clean) - set(present_students)))
st.sidebar.metric("Remaining", len(absentees))
if absentees:
    st.sidebar.dataframe(pd.DataFrame(absentees, columns=["Name"]), hide_index=True)
else:
    st.sidebar.success("All Present")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.caption("Turn your head slowly to verify & mark attendance.")
    webrtc_streamer(
        key="locus-safe",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=BlinkProcessor,
        media_stream_constraints={"video": {"width": 1024, "height": 720}, "audio": False},
        async_processing=True,
    )
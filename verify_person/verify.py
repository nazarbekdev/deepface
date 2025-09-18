import os
import cv2
import sqlite3
from datetime import datetime
from deepface import DeepFace
import pandas as pd
from playsound import playsound

# ---------- CONFIG ----------
KNOWN_FACES_DIR = "known_faces"
DB_FILE = "employees.db"
CAMERA_IDX = 0
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"
USE_THRESHOLD_FROM_DF = True
FALLBACK_THRESHOLD = 0.45
# -----------------------------

person_data = {
    "Nazarbek": {"full_name": "Nazarbek Qobulov", "class": "9-sinf"},
    "Azizbek": {"full_name": "Azizbek X.", "class": "10-sinf"}
}

# --------- SQLite yordamchi funktsiyalar ----------
def init_db(db_file: str):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        folder_name TEXT UNIQUE,
        full_name TEXT,
        class TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS access_logs (
        id INTEGER PRIMARY KEY,
        folder_name TEXT,
        full_name TEXT,
        timestamp TEXT,
        matched INTEGER,
        distance REAL
    )
    """)
    conn.commit()
    return conn

def seed_employees(conn, mapping):
    cur = conn.cursor()
    for folder_name, info in mapping.items():
        cur.execute("""
            INSERT OR IGNORE INTO employees (folder_name, full_name, class)
            VALUES (?, ?, ?)
        """, (folder_name, info.get("full_name", folder_name), info.get("class", "")))
    conn.commit()

def get_employee(conn, folder_name):
    cur = conn.cursor()
    cur.execute("SELECT folder_name, full_name, class FROM employees WHERE folder_name = ?", (folder_name,))
    row = cur.fetchone()
    if row:
        return {"folder_name": row[0], "full_name": row[1], "class": row[2]}
    return None

def log_access(conn, folder_name, full_name, matched, distance):
    cur = conn.cursor()
    ts = datetime.now().isoformat()
    cur.execute("""
        INSERT INTO access_logs (folder_name, full_name, timestamp, matched, distance)
        VALUES (?, ?, ?, ?, ?)
    """, (folder_name, full_name, ts, 1 if matched else 0, float(distance) if distance is not None else None))
    conn.commit()
# --------------------------------------------------

def safe_get_top_result(df_list):
    if not df_list or len(df_list) == 0:
        return None
    df = df_list[0]
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        return df.iloc[0].to_dict()
    return None

def main():
    conn = init_db(DB_FILE)
    seed_employees(conn, person_data)

    cap = cv2.VideoCapture(CAMERA_IDX)
    if not cap.isOpened():
        print("Kamera ochilmadi.")
        return

    print("Kamera ishlamoqda. 'q' bilan chiqish.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            df_list = DeepFace.find(
                img_path=frame, db_path=KNOWN_FACES_DIR,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                silent=True,
            )
        except Exception as e:
            print("DeepFace.find xatosi:", e)
            df_list = None

        top = safe_get_top_result(df_list)
        if top is not None:
            identity_path = top.get("identity", "")
            person_name = os.path.basename(os.path.dirname(identity_path)) if identity_path else None
            distance = top.get("distance", None)
            threshold = top.get("threshold", None)

            if USE_THRESHOLD_FROM_DF and threshold is not None:
                matched = float(distance) <= float(threshold)
            else:
                matched = float(distance) <= FALLBACK_THRESHOLD if distance is not None else False

            if matched:
                emp = get_employee(conn, person_name) or {"folder_name": person_name, "full_name": person_name, "class": ""}
                text1 = f"Ism: {emp['full_name']}"
                text2 = f"Sinf: {emp['class']}"
                color = (0, 255, 0)
                log_access(conn, person_name, emp['full_name'], True, distance)
                playsound("access.mp3")   # ✅ topildi
            else:
                text1 = "Sizni tanib bo'lmadi"
                text2 = f"Distance: {distance:.3f}" if distance is not None else ""
                color = (0, 0, 255)
                log_access(conn, None, "Unknown", False, distance)
                playsound("error.mp3")

            try:
                sx = int(top.get("source_x", 50))
                sy = int(top.get("source_y", 50))
                sw = int(top.get("source_w", 200))
                sh = int(top.get("source_h", 200))
                cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), color, 2)
                cv2.putText(frame, text1, (sx, sy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, text2, (sx, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except Exception:
                cv2.putText(frame, text1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.putText(frame, text2, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        else:
            cv2.putText(frame, "Sizni tanib bo'lmadi", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            log_access(conn, None, "Unknown", False, None)
            playsound("error.mp3")

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == "__main__":
    main()


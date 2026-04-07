"""
flask_resume_pipeline/app.py

Resume Processing Pipeline with Background Monitoring

Flask app that monitors Google Drive folder for PDF/DOCX resumes, extracts text,
parses with Gemini API, stores in MySQL with duplicate detection.

Database: MySQL 8.0+
Services: Google Drive API, Gemini API
"""

import io
import json
import logging
import os
import threading
import time
from datetime import datetime
from urllib.parse import urlparse

import pdfplumber
import mysql.connector
import google.generativeai as genai
from docx import Document
from flask import Flask, jsonify
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv

load_dotenv()

# ── Application Setup ─────────────────────────────────────────────────────────

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Environment Configuration ─────────────────────────────────────────────────

GDRIVE_FOLDER_ID     = os.environ["GDRIVE_FOLDER_ID"]
SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json")
GEMINI_API_KEYS      = [k.strip() for k in os.environ.get("GEMINI_API_KEYS", "").split(",") if k.strip()]
GEMINI_MODEL         = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
POLL_INTERVAL        = int(os.environ.get("POLL_INTERVAL_SECONDS", "300"))
PROCESSED_IDS_FILE   = os.environ.get("PROCESSED_IDS_FILE", "processed_ids.txt")

# Parse DATABASE_URL — format: mysql://user:password@host:port/database
_db_url = urlparse(os.environ["DATABASE_URL"].replace("mysql://", "mysql://"))
DB_HOST = _db_url.hostname
DB_PORT = _db_url.port or 3306
DB_USER = _db_url.username
DB_PASSWORD = _db_url.password
DB_NAME = _db_url.path.lstrip("/")

# ── Thread-Safe State Management ───────────────────────────────────────────────

_state_lock = threading.Lock()
_state = {
    "status":         "idle",
    "last_run":       None,
    "last_run_found": 0,
    "total_processed": 0,
    "errors":         [],
    "log":            [],
}

def _log(msg: str, level: str = "info"):
    """Thread-safe logging to logger and state (max 200 entries).""" 
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    getattr(log, level)(msg)
    with _state_lock:
        _state["log"].append(entry)
        if len(_state["log"]) > 200:
            _state["log"] = _state["log"][-200:]

# ── Google Drive Integration ───────────────────────────────────────────────────

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def get_drive_service():
    """Returns authenticated Google Drive API service using service account."""
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_JSON, scopes=DRIVE_SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def list_drive_files(service):
    """Lists all PDF/DOCX files from configured Drive folder (paginated)."""
    query = (
        f"'{GDRIVE_FOLDER_ID}' in parents and trashed = false "
        "and (mimeType='application/pdf' or "
        "mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document')"
    )
    results, page_token = [], None
    while True:
        resp = service.files().list(
            q=query, spaces="drive", pageToken=page_token,
            fields="nextPageToken, files(id, name, mimeType, createdTime)"
        ).execute()
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results

def download_file(service, file_id: str) -> bytes:
    """Downloads file from Drive using chunked download."""
    req = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl  = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()

# ── Processed Resume Tracking (Layer 1: Local File) ───────────────────────────

def load_processed_ids() -> set:
    """Loads processed file IDs from local disk."""
    from pathlib import Path
    p = Path(PROCESSED_IDS_FILE)
    return set(p.read_text().splitlines()) if p.exists() else set()

def save_processed_id(file_id: str):
    """Appends file ID to processed list."""
    with open(PROCESSED_IDS_FILE, "a") as f:
        f.write(file_id + "\n")

# ── Duplicate Detection & Database Validation (Layer 2: Database) ──────────────

def is_already_in_db(file_id: str) -> bool:
    """Fallback check — verifies file exists in DB (handles missing txt file)."""
    try:
        cv_url = f"https://drive.google.com/file/d/{file_id}/view"
        conn = get_db_conn()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute(
                "SELECT id FROM candidates WHERE cv_url = %s LIMIT 1",
                (cv_url,)
            )
            result = cur.fetchone()
            cur.close()
            return result is not None
        finally:
            conn.close()
    except Exception as e:
        _log(f"DB validation check failed: {e} - using file check only", "warning")
        return False

def is_duplicate_candidate(email: str, phone: str) -> bool:
    """Checks if candidate with same email/phone already exists in DB."""
    if not email and not phone:
        return False
    try:
        conn = get_db_conn()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute(
                """SELECT id FROM candidates
                   WHERE (email = %s AND email IS NOT NULL AND email != '')
                      OR (phone = %s AND phone IS NOT NULL AND phone != '')
                   LIMIT 1""",
                (email, phone)
            )
            result = cur.fetchone()
            cur.close()
            return result is not None
        finally:
            conn.close()
    except Exception as e:
        _log(f"Duplicate candidate check failed: {e}", "warning")
        return False

def sync_processed_ids_from_db():
    """Rebuilds processed_ids.txt from DB on startup (recovery mechanism)."""
    try:
        conn = get_db_conn()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT cv_url FROM candidates WHERE cv_url IS NOT NULL")
            rows = cur.fetchall()
            cur.close()
        finally:
            conn.close()

        db_ids = set()
        for row in rows:
            url = row.get("cv_url", "")
            if "/file/d/" in url:
                fid = url.split("/file/d/")[1].split("/")[0]
                db_ids.add(fid)

        if not db_ids:
            _log("Sync: No records in DB yet — nothing to restore.")
            return

        local_ids = load_processed_ids()
        missing   = db_ids - local_ids

        if missing:
            _log(f"Sync: {len(missing)} IDs in DB but missing from processed_ids.txt — restoring...")

        all_ids = local_ids | db_ids
        with open(PROCESSED_IDS_FILE, "w") as f:
            for fid in sorted(all_ids):
                f.write(fid + "\n")

        if missing:
            _log(f"Sync complete — {len(all_ids)} unique IDs in processed_ids.txt.")
        else:
            _log(f"Sync: already in sync — {len(all_ids)} unique IDs.")

    except Exception as e:
        _log(f"Startup DB sync failed: {e}", "warning")

# ── Document Text Extraction ───────────────────────────────────────────────────

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extracts text from PDF or DOCX file."""
    text = ""
    try:
        if filename.lower().endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        elif filename.lower().endswith(".docx"):
            doc = Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        _log(f"Text extraction error ({filename}): {e}", "error")
    return text

# ── Resume Parsing with Google Gemini API ──────────────────────────────────────

PROMPT = """
You are an expert Resume Parser. Extract the following details from the resume text and return them in a strict JSON format.

Fields to extract:
1. Name (Full name)
2. Phone (Mobile number)
3. Email (Email address)
4. Salary (Current salary as a string, e.g., "12 LPA", default "0")
5. Expected CTC (Expected salary as a string, default "0")
6. Notice (Notice period in days as a string, default "0")
7. Total Experience Years (Experience in years as a float/number, e.g., 2.5)
8. Location (Current city/state)
9. Current Company Name (The company where the candidate is currently working)
10. Skills (List/Array of technical and soft skills)
11. Previous Companies Name (List/Array of names of companies worked at previously)
12. Education (Highest degree and university)
13. Job Title (Current or most recent designation)
14. Company Names (List/Array of ALL companies mentioned in the resume)

Return ONLY raw JSON. No markdown. No backticks.
Keys to use: 
name, phone, email, salary, expected_ctc, notice, total_experience_years, 
location, current_company_name, skills, previous_companies_name, 
education, job_title, company_names

Resume Text:
{text}
"""

def parse_with_gemini(text: str) -> dict:
    """Parses resume with Gemini API; auto-rotates keys on quota exhaustion."""
    last_error = None
    for idx, key in enumerate(GEMINI_API_KEYS):
        try:
            genai.configure(api_key=key)
            model  = genai.GenerativeModel(GEMINI_MODEL)
            raw    = model.generate_content(PROMPT.format(text=text)).text
            clean  = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        except json.JSONDecodeError as e:
            _log(f"Gemini JSON error (key {idx+1}): {e}", "error")
            return {}
        except Exception as e:
            if any(kw in str(e).lower() for kw in ("quota", "rate", "exceeded")):
                _log(f"Gemini key {idx+1} quota exhausted, rotating...", "warning")
                last_error = e
                continue
            raise
    raise RuntimeError(f"All Gemini keys exhausted: {last_error}")

# ── MySQL Database Operations ──────────────────────────────────────────────────

def get_db_conn():
    """Creates and returns a MySQL connection."""
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset="utf8mb4",
        autocommit=False,
        use_pure=True,
    )

def insert_candidate(parsed: dict, cv_url: str):
    """Inserts parsed candidate into DB; converts list fields to comma-separated strings."""

    skills_list = parsed.get("skills", [])
    skills_str = ", ".join(skills_list) if isinstance(skills_list, list) else str(skills_list)

    prev_companies_list = parsed.get("previous_companies_name", [])
    prev_comp_str = ", ".join(prev_companies_list) if isinstance(prev_companies_list, list) else str(prev_companies_list)

    all_companies_list = parsed.get("company_names", [])
    all_comp_str = ", ".join(all_companies_list) if isinstance(all_companies_list, list) else str(all_companies_list)

    sql = """
        INSERT INTO candidates 
        (name, phone, email, salary, expected_ctc, notice, total_experience_years, 
         location, cv_url, current_company_name, skills, previous_companies_name, 
         education, job_title, company_names, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    """

    values = (
        parsed.get("name"),
        parsed.get("phone"),
        parsed.get("email"),
        str(parsed.get("salary", "0")),
        str(parsed.get("expected_ctc", "0")),
        str(parsed.get("notice", "0")),
        float(parsed.get("total_experience_years", 0.0)),
        parsed.get("location"),
        cv_url,
        parsed.get("current_company_name"),
        skills_str,
        prev_comp_str,
        parsed.get("education"),
        parsed.get("job_title"),
        all_comp_str
    )

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, values)
        conn.commit()
        cur.close()
    finally:
        conn.close()

# ── Core Pipeline Logic ────────────────────────────────────────────────────────

def run_pipeline():
    """
    Main pipeline: Drive → Extract → Parse → DB → Mark processed.

    Layers:
    1. Local file check (processed_ids.txt)
    2. DB validation (fallback if txt missing)
    3. Duplicate candidate check (email/phone)
    """
    _log("=== Pipeline run started ===")
    with _state_lock:
        _state["status"] = "running"
        _state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    found = 0
    try:
        drive_svc  = get_drive_service()
        all_files  = list_drive_files(drive_svc)
        processed  = load_processed_ids()

        # Layer 1: Fast local file check
        new_files = [f for f in all_files if f["id"] not in processed]

        # Layer 2: DB validation
        validated_new = []
        for f in new_files:
            if is_already_in_db(f["id"]):
                _log(f"Already in DB (not in txt): {f['name']} — marking as processed")
                save_processed_id(f["id"])
            else:
                validated_new.append(f)

        _log(f"Drive: {len(all_files)} total | {len(processed)} in txt | {len(validated_new)} truly new")

        for file_meta in validated_new:
            fid, fname = file_meta["id"], file_meta["name"]
            _log(f"Processing: {fname}")
            try:
                file_bytes = download_file(drive_svc, fid)
                text       = extract_text(file_bytes, fname)

                if not text.strip():
                    _log(f"No text in {fname}, skipping", "warning")
                    save_processed_id(fid)
                    continue

                parsed = parse_with_gemini(text)
                if not parsed:
                    _log(f"Gemini returned empty for {fname}", "warning")
                    continue

                # Layer 3: Duplicate candidate check
                email = parsed.get("email", "")
                phone = parsed.get("phone", "")
                if is_duplicate_candidate(email, phone):
                    _log(f"Duplicate candidate detected ({email or phone}) — skipping {fname}", "warning")
                    save_processed_id(fid)
                    continue

                cv_url = f"https://drive.google.com/file/d/{fid}/view"
                insert_candidate(parsed, cv_url)
                save_processed_id(fid)
                found += 1
                _log(f"Saved: {parsed.get('name', 'Unknown')}")

            except Exception as e:
                _log(f"Error on {fname}: {e}", "error")
                with _state_lock:
                    _state["errors"].append({"file": fname, "error": str(e), "time": datetime.now().strftime("%H:%M:%S")})

    except Exception as e:
        _log(f"Pipeline error: {e}", "error")

    with _state_lock:
        _state["last_run_found"]   = found
        _state["total_processed"] += found

    _log(f"=== Run complete — {found} new resume(s) saved ===")

# ── Background Polling Thread ──────────────────────────────────────────────────

def polling_thread():
    """Runs pipeline every POLL_INTERVAL seconds in a loop."""
    _log(f"Background poller started (interval={POLL_INTERVAL}s)")
    while True:
        run_pipeline()
        with _state_lock:
            _state["status"] = "sleeping"
        _log(f"Sleeping {POLL_INTERVAL}s until next check...")
        time.sleep(POLL_INTERVAL)
        with _state_lock:
            _state["status"] = "running"

# ── Flask REST API Endpoints ───────────────────────────────────────────────────

@app.route("/")
def index():
    """Health check endpoint."""
    return jsonify({
        "app":     "Resume Pipeline",
        "status":  "alive",
        "message": "Background poller is active. Use /status for details.",
    })

@app.route("/status")
def status():
    """Returns pipeline status and metrics."""
    with _state_lock:
        return jsonify({
            "status":          _state["status"],
            "last_run":        _state["last_run"],
            "last_run_found":  _state["last_run_found"],
            "total_processed": _state["total_processed"],
            "recent_errors":   _state["errors"][-5:],
        })

@app.route("/logs")
def logs():
    """Returns last 50 log entries."""
    with _state_lock:
        return jsonify({"logs": _state["log"][-50:]})

@app.route("/run", methods=["POST"])
def trigger_run():
    """Manually triggers pipeline run in background thread."""
    t = threading.Thread(target=run_pipeline, daemon=True)
    t.start()
    return jsonify({"message": "Pipeline triggered manually."}), 202

# ── Application Startup Initialization ─────────────────────────────────────────

def start_background_poller():
    """Syncs processed IDs from DB then starts polling thread."""
    sync_processed_ids_from_db()
    t = threading.Thread(target=polling_thread, daemon=True)
    t.start()
    log.info("Background polling thread started.")

with app.app_context():
    start_background_poller()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
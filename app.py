"""
flask_resume_pipeline/app.py

Resume Processing Pipeline with Background Monitoring

Flow:
Google Drive -> Extract -> Gemini parse -> Save details in MySQL
AND save original resume in local resumes/files folder with unique id.
Store backend preview URL in candidates.cv_url
"""

import io
import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import google.generativeai as genai
import mysql.connector
import pdfplumber
from docx import Document
from dotenv import load_dotenv
from flask import Flask, jsonify, send_file
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

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

GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")
if not GDRIVE_FOLDER_ID:
    raise ValueError("Missing required env var: GDRIVE_FOLDER_ID")

SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json")
GEMINI_API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", "120"))
PROCESSED_IDS_FILE = os.getenv("PROCESSED_IDS_FILE", "processed_ids.txt")

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "").rstrip("/")
RESUME_STORAGE_DIR = Path(os.getenv("RESUME_STORAGE_DIR", "resumes/files"))
RESUME_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("Missing required env var: DATABASE_URL")

_db_url = urlparse(DATABASE_URL.replace("mysql://", "mysql://"))
DB_HOST = _db_url.hostname
DB_PORT = _db_url.port or 3306
DB_USER = _db_url.username
DB_PASSWORD = _db_url.password
DB_NAME = _db_url.path.lstrip("/")

# ── Thread-Safe State Management ──────────────────────────────────────────────

_state_lock = threading.Lock()
_state = {
    "status": "idle",
    "last_run": None,
    "last_run_found": 0,
    "total_processed": 0,
    "errors": [],
    "log": [],
}


def _log(msg: str, level: str = "info"):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    getattr(log, level)(msg)
    with _state_lock:
        _state["log"].append(entry)
        if len(_state["log"]) > 200:
            _state["log"] = _state["log"][-200:]


# ── Safe converters for DB insertion ──────────────────────────────────────────

def to_text(v):
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def to_float(v, default=0.0):
    try:
        if isinstance(v, dict):
            for k in ("value", "years", "total", "experience"):
                if k in v:
                    return float(v[k])
            return default
        return float(str(v).strip())
    except Exception:
        return default


# ── Resume file helpers ────────────────────────────────────────────────────────

def ext_from_filename(filename: str) -> str:
    fn = (filename or "").lower()
    if fn.endswith(".pdf"):
        return ".pdf"
    if fn.endswith(".docx"):
        return ".docx"
    return ".bin"


def save_resume_locally(file_bytes: bytes, original_name: str):
    file_id = uuid.uuid4().hex
    ext = ext_from_filename(original_name)
    stored_name = f"{file_id}{ext}"
    out_path = RESUME_STORAGE_DIR / stored_name
    with open(out_path, "wb") as f:
        f.write(file_bytes)
    return file_id, stored_name


def build_resume_url(file_id: str) -> str:
    if BACKEND_BASE_URL:
        return f"{BACKEND_BASE_URL}/resume/{file_id}"
    return f"/resume/{file_id}"


# ── Google Drive Integration ───────────────────────────────────────────────────

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_JSON, scopes=DRIVE_SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_drive_files(service):
    query = (
        f"'{GDRIVE_FOLDER_ID}' in parents and trashed = false "
        "and (mimeType='application/pdf' or "
        "mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document')"
    )
    results, page_token = [], None
    while True:
        resp = service.files().list(
            q=query,
            spaces="drive",
            pageToken=page_token,
            fields="nextPageToken, files(id, name, mimeType, createdTime)",
        ).execute()
        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return results


def download_file(service, file_id: str) -> bytes:
    req = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()


# ── Processed Resume Tracking ──────────────────────────────────────────────────

def load_processed_ids() -> set:
    p = Path(PROCESSED_IDS_FILE)
    return set(p.read_text().splitlines()) if p.exists() else set()


def save_processed_id(file_id: str):
    with open(PROCESSED_IDS_FILE, "a") as f:
        f.write(file_id + "\n")


# ── Duplicate Detection ────────────────────────────────────────────────────────

def is_already_in_db(file_id: str) -> bool:
    """Checks old Drive URL existence in DB to avoid reprocessing."""
    try:
        drive_url = f"https://drive.google.com/file/d/{file_id}/view"
        conn = get_db_conn()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute("SELECT id FROM candidates WHERE cv_url = %s LIMIT 1", (drive_url,))
            result = cur.fetchone()
            cur.close()
            return result is not None
        finally:
            conn.close()
    except Exception as e:
        _log(f"DB validation check failed: {e}", "warning")
        return False


def is_duplicate_candidate(email: str, phone: str) -> bool:
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
                (email, phone),
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
    """Optional recovery sync using old Drive-style cv_url only."""
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

        local_ids = load_processed_ids()
        all_ids = local_ids | db_ids

        with open(PROCESSED_IDS_FILE, "w") as f:
            for fid in sorted(all_ids):
                f.write(fid + "\n")

        _log(f"Sync complete — {len(all_ids)} IDs in processed_ids.txt.")
    except Exception as e:
        _log(f"Startup DB sync failed: {e}", "warning")


# ── Document Text Extraction ───────────────────────────────────────────────────

def extract_text(file_bytes: bytes, filename: str) -> str:
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


# ── Gemini Parsing ─────────────────────────────────────────────────────────────

PROMPT = """
 You are an expert Resume Parser. Extract the following details from the resume text below:
    
    1. Name
    2. Email
    3. Phone Number
    4. Total Experience (a number)
    5. Skills (JSON array of strings)
    6. Current Company
    7. Job Title (Current or Last)
    8. Education (Highest Degree)
    9. Current Location
    10. Current Salary (If mentioned, else 0)
    11. Expected Salary (If mentioned, else 0)
    12. Notice Period (In days, if mentioned, else 0)

    Return ONLY raw JSON. No markdown formatting.
    Keys: name, email, phone, totalExperienceYears, skills, currentCompanyName, jobTitle, education, location, salary, expected_ctc, notice

    Resume Text:
    {text}
"""


def parse_with_gemini(text: str) -> dict:
    if not GEMINI_API_KEYS:
        _log("No GEMINI_API_KEYS configured", "error")
        return {}

    last_error = None
    for idx, key in enumerate(GEMINI_API_KEYS):
        try:
            _log(f"Trying Gemini key {idx + 1}/{len(GEMINI_API_KEYS)}")
            genai.configure(api_key=key)
            model = genai.GenerativeModel(GEMINI_MODEL)
            raw = model.generate_content(PROMPT.format(text=text[:15000])).text
            clean = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(clean)
            _log(f"Gemini key {idx + 1} succeeded")
            return result

        except json.JSONDecodeError as e:
            _log(f"Gemini JSON parse error (key {idx + 1}): {e}", "error")
            return {}

        except Exception as e:
            msg = str(e).lower()
            if "quota" in msg or "rate" in msg or "exceeded" in msg or "429" in msg:
                wait_s = 60
                _log(
                    f"Gemini key {idx + 1} quota/rate limit hit — waiting {wait_s}s before trying next key...",
                    "warning",
                )
                last_error = e
                time.sleep(wait_s)
                continue
            # Non-quota error — don't rotate, just fail fast
            _log(f"Gemini unexpected error (key {idx + 1}): {e}", "error")
            raise

    # All keys exhausted — skip this resume, retry next poll cycle
    _log(
        f"All {len(GEMINI_API_KEYS)} Gemini key(s) exhausted — skipping resume, will retry next poll cycle.",
        "warning",
    )
    return {}


# ── MySQL ──────────────────────────────────────────────────────────────────────

def get_db_conn():
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
    skills_list = parsed.get("skills", [])
    skills_str = ", ".join(skills_list) if isinstance(skills_list, list) else to_text(skills_list)

    prev_companies_list = parsed.get("previous_companies_name", [])
    prev_comp_str = ", ".join(prev_companies_list) if isinstance(prev_companies_list, list) else to_text(prev_companies_list)

    all_companies_list = parsed.get("company_names", [])
    all_comp_str = ", ".join(all_companies_list) if isinstance(all_companies_list, list) else to_text(all_companies_list)

    sql = """
        INSERT INTO candidates
        (name, phone, email, salary, expected_ctc, notice, total_experience_years,
         location, cv_url, current_company_name, skills, previous_companies_name,
         education, job_title, company_names, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
    """

    values = (
        to_text(parsed.get("name")),
        to_text(parsed.get("phone")),
        to_text(parsed.get("email")),
        to_text(parsed.get("salary", "0")),
        to_text(parsed.get("expected_ctc", "0")),
        to_text(parsed.get("notice", "0")),
        to_float(parsed.get("total_experience_years", 0.0)),
        to_text(parsed.get("location")),
        cv_url,
        to_text(parsed.get("current_company_name")),
        skills_str,
        prev_comp_str,
        to_text(parsed.get("education")),
        to_text(parsed.get("job_title")),
        all_comp_str,
    )

    conn = get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, values)
        conn.commit()
        cur.close()
    finally:
        conn.close()


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline():
    _log("=== Pipeline run started ===")
    with _state_lock:
        _state["status"] = "running"
        _state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    found = 0
    try:
        drive_svc = get_drive_service()
        all_files = list_drive_files(drive_svc)
        processed = load_processed_ids()

        new_files = [f for f in all_files if f["id"] not in processed]

        validated_new = []
        for f in new_files:
            if is_already_in_db(f["id"]):
                _log(f"Already in DB (old drive cv_url): {f['name']} — marking processed")
                save_processed_id(f["id"])
            else:
                validated_new.append(f)

        _log(f"Drive: {len(all_files)} total | {len(processed)} in txt | {len(validated_new)} truly new")

        for file_meta in validated_new:
            fid, fname = file_meta["id"], file_meta["name"]
            _log(f"Processing: {fname}")
            try:
                file_bytes = download_file(drive_svc, fid)
                text = extract_text(file_bytes, fname)

                if not text.strip():
                    _log(f"No text in {fname}, skipping", "warning")
                    save_processed_id(fid)
                    continue

                parsed = parse_with_gemini(text)
                if not parsed:
                    _log(f"Gemini returned empty for {fname} — will retry next cycle", "warning")
                    # Do NOT save_processed_id here so it retries next poll
                    continue

                email = to_text(parsed.get("email", "")) or ""
                phone = to_text(parsed.get("phone", "")) or ""
                if is_duplicate_candidate(email, phone):
                    _log(f"Duplicate candidate ({email or phone}) — skipping {fname}", "warning")
                    save_processed_id(fid)
                    continue

                unique_file_id, stored_name = save_resume_locally(file_bytes, fname)
                cv_url = build_resume_url(unique_file_id)

                insert_candidate(parsed, cv_url)
                save_processed_id(fid)
                found += 1
                _log(f"Saved: {to_text(parsed.get('name', 'Unknown'))} | file={stored_name} | cv_url={cv_url}")

                time.sleep(15)

            except Exception as e:
                _log(f"Error on {fname}: {e}", "error")
                with _state_lock:
                    _state["errors"].append({"file": fname, "error": str(e), "time": datetime.now().strftime("%H:%M:%S")})

    except Exception as e:
        _log(f"Pipeline error: {e}", "error")

    with _state_lock:
        _state["last_run_found"] = found
        _state["total_processed"] += found

    _log(f"=== Run complete — {found} new resume(s) saved ===")


# ── Poller ─────────────────────────────────────────────────────────────────────

def polling_thread():
    _log(f"Background poller started (interval={POLL_INTERVAL}s)")
    while True:
        run_pipeline()
        with _state_lock:
            _state["status"] = "sleeping"
        _log(f"Sleeping {POLL_INTERVAL}s until next check...")
        time.sleep(POLL_INTERVAL)
        with _state_lock:
            _state["status"] = "running"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return jsonify(
        {
            "app": "Resume Pipeline",
            "status": "alive",
            "message": "Background poller is active. Use /status for details.",
        }
    )


@app.route("/status")
def status():
    with _state_lock:
        return jsonify(
            {
                "status": _state["status"],
                "last_run": _state["last_run"],
                "last_run_found": _state["last_run_found"],
                "total_processed": _state["total_processed"],
                "recent_errors": _state["errors"][-5:],
            }
        )


@app.route("/logs")
def logs():
    with _state_lock:
        return jsonify({"logs": _state["log"][-50:]})


@app.route("/run", methods=["POST"])
def trigger_run():
    t = threading.Thread(target=run_pipeline, daemon=True)
    t.start()
    return jsonify({"message": "Pipeline triggered manually."}), 202


@app.route("/resume/<string:file_id>")
def resume_preview(file_id):
    matches = list(RESUME_STORAGE_DIR.glob(f"{file_id}.*"))
    if not matches:
        return jsonify({"error": "Resume not found"}), 404
    return send_file(matches[0], as_attachment=False)


# ── Startup ────────────────────────────────────────────────────────────────────

def start_background_poller():
    sync_processed_ids_from_db()
    t = threading.Thread(target=polling_thread, daemon=True)
    t.start()
    log.info("Background polling thread started.")


with app.app_context():
    start_background_poller()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
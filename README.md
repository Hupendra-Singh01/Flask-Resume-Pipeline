# Flask Resume Pipeline

Flask-based resume processing pipeline that monitors a Google Drive folder for new PDF/DOCX resumes, extracts text, parses candidate details using the Gemini API, and stores structured results in MySQL with duplicate detection.

## Features
- Google Drive folder monitoring (polling)
- PDF/DOCX text extraction
- Gemini API parsing (key rotation support)
- MySQL storage + duplicate checks (email/phone + file tracking)
- REST endpoints: `/`, `/status`, `/logs`, `/run`

## Setup
1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment variables in `.env` (see `.env.template`)
3. Run:
   ```bash
   python app.py
   ```

## Notes
- Do not commit secrets like `service_account.json` or `.env`.
- Do not commit `venv/` (keep it in `.gitignore`).

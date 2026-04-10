import os


def build_resume_url(file_id: str) -> str:
    """
    Build resume preview URL using unique file_id stored in resumes/files.

    
    """
    base_url = os.getenv("BACKEND_BASE_URL", "").strip().rstrip("/")
    path = f"/resume/{file_id}"

    if base_url:
        return f"{base_url}{path}"
    return path
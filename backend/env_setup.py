# env_setup.py
# MUST BE IMPORTED FIRST - Sets environment variables before any other imports
import os

# Disable HuggingFace symlinks (Windows compatibility)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"


# uvicorn app:app --host 0.0.0.0 --port 8000 --env-file .env --reload
# # ..\.venv\Scripts\Activate.ps1
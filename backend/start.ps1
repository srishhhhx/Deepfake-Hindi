# start.ps1
# Startup script for backend with Windows symlink fix

# Set environment variables to disable HuggingFace symlinks
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_DISABLE_SYMLINKS = "1"

Write-Host "Starting server with HF_HUB_DISABLE_SYMLINKS=1..." -ForegroundColor Green

# Run uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --env-file .env --reload

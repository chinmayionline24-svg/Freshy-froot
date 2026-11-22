FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# Install CPU-only PyTorch to save space and prevent crashes
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

# Start the server (Render automatically sets the $PORT)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

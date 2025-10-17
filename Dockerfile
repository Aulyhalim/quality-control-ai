# 1. Gunakan base image Python yang ringan
FROM python:3.10-slim

# 2. Set direktori kerja di dalam container
WORKDIR /app

# 3. Instal dependensi sistem yang dibutuhkan oleh OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Salin file requirements dan instal pustaka Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Salin seluruh kode proyek ke dalam container
COPY . .

# --- PERBAIKAN UTAMA DI SINI ---
# 6. Set PYTHONPATH agar Python bisa menemukan module 'app'
ENV PYTHONPATH=/app

# 7. Buka port yang digunakan oleh aplikasi
EXPOSE 8000
EXPOSE 7860

# 8. Perintah default untuk menjalankan API saat container dimulai
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
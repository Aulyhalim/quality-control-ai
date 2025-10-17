# main.py

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import io

# --- BARU: Impor semua dependensi model dari satu tempat ---
from app.model_loader import model_for_pred as model, transform, class_names

# --- 1. Inisialisasi Aplikasi FastAPI ---
app = FastAPI(
    title="API Kontrol Kualitas Produk",
    description="API ini menerima gambar produk dan mengklasifikasikannya sebagai 'good' atau 'defective'.",
    version="2.0.0" # Versi diperbarui
)

# --- 2. Buat Endpoint Prediksi ---
@app.post("/inspect/", summary="Klasifikasi Gambar Produk")
async def inspect_image(file: UploadFile = File(..., description="File gambar yang akan diinspeksi.")):
    """
    Endpoint ini menerima file gambar, memprosesnya, dan mengembalikan
    hasil prediksi (defective/good) beserta confidence score.
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item()

    return {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": f"{confidence_score:.2f}"
    }

# --- 3. Buat Endpoint Root (Homepage) ---
@app.get("/", summary="Endpoint Root")
def read_root():
    return {"message": "Selamat datang di API Kontrol Kualitas v2.0. Buka /docs untuk mencoba."}
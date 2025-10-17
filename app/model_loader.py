# app/model_loader.py

import torch
from torchvision import models, transforms
from pathlib import Path

# --- PERBAIKAN: Path dinamis yang bekerja dari mana saja ---
BASE_DIR = Path(__file__).parent.parent  # Naik 1 level dari app/ ke root project/
QUANTIZED_MODEL_PATH = BASE_DIR / 'models' / 'quality_control_model_quantized.ptq'
ORIGINAL_MODEL_PATH = BASE_DIR / 'models' / 'quality_control_model.pth'
NUM_CLASSES = 2
CLASS_NAMES = ['defective', 'good']

def load_all_models_and_deps():
    """
    Memuat SEMUA model dan dependensi yang dibutuhkan oleh aplikasi.
    Satu-satunya sumber kebenaran untuk model.
    """
    # --- 1. Muat Model Terkuantisasi (Untuk Prediksi Cepat) ---
    print(f"Loading quantized model from: {QUANTIZED_MODEL_PATH}")
    quantized_model = torch.jit.load(str(QUANTIZED_MODEL_PATH), map_location='cpu')
    quantized_model.eval()

    # --- 2. Muat Model Asli (Untuk Grad-CAM) ---
    print(f"Loading original model from: {ORIGINAL_MODEL_PATH}")
    original_model = models.resnet18()
    num_ftrs = original_model.fc.in_features
    original_model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
    original_model.load_state_dict(torch.load(str(ORIGINAL_MODEL_PATH), map_location='cpu'))
    original_model.eval()

    # --- 3. Definisikan Transformasi Gambar ---
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return quantized_model, original_model, image_transform, CLASS_NAMES

# --- Muat sekali saat aplikasi dimulai ---
# Beri nama yang jelas untuk setiap variabel
model_for_pred, model_for_cam, transform, class_names = load_all_models_and_deps()

print("✅ All models, transform, and class names loaded successfully from model_loader.py")
# scripts/05_evaluate_on_test_set.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import yaml
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- PERBAIKAN DI SINI ---
# Impor nama variabel yang benar: 'model_for_pred' bukan 'model'
from app.model_loader import model_for_pred, transform, class_names

def evaluate_on_test_set():
    """Mengevaluasi model final pada test set yang tersembunyi."""
    print("--- Memulai Evaluasi Final pada Test Set ---")

    # 1. Muat Konfigurasi
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("[ERROR] File 'config.yaml' tidak ditemukan. Pastikan Anda menjalankan skrip dari direktori utama proyek.")
        return
    
    DATA_DIR = config['data_dir']
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    OUTPUT_DIR = os.path.join(config['output_dir'], 'metrics')
    
    if not os.path.exists(TEST_DIR) or not os.listdir(TEST_DIR):
        print(f"\n[ERROR] Direktori test set tidak ditemukan atau kosong di '{TEST_DIR}'.")
        print("Pastikan Anda sudah membuat folder 'test' dan memindahkan sebagian data validasi ke dalamnya.")
        return

    # 2. Muat Test Dataset
    print(f"Memuat test dataset dari: {TEST_DIR}")
    try:
        test_dataset = datasets.ImageFolder(TEST_DIR, transform)
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        print(f"Ditemukan {len(test_dataset)} gambar dalam test set.")
    except Exception as e:
        print(f"[ERROR] Gagal memuat dataset: {e}")
        return

    # 3. Lakukan Prediksi
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --- PERBAIKAN DI SINI ---
    # Gunakan variabel yang benar: 'model_for_pred'
    model_to_evaluate = model_for_pred.to(device)
    model_to_evaluate.eval()

    all_labels = []
    all_preds = []

    print("\nMemulai prediksi pada test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # --- PERBAIKAN DI SINI ---
            outputs = model_to_evaluate(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 4. Tampilkan Laporan Klasifikasi
    print("\n--- Laporan Klasifikasi Final (Test Set) ---")
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)

    # Simpan laporan ke file teks
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'final_evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("Laporan Kinerja Model pada Test Set\n")
        f.write("=======================================\n\n")
        f.write(report)
    print(f"\nLaporan disimpan ke: '{report_path}'")

    # 5. Buat dan Simpan Confusion Matrix
    print("\nMembuat confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Kinerja pada Test Set')
    
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix disimpan di: '{cm_path}'")
    print("\n--- Evaluasi Selesai ---")

if __name__ == '__main__':
    try:
        import seaborn
    except ImportError:
        print("\n[PERHATIAN] Pustaka 'seaborn' belum terinstal. Silakan jalankan: pip install seaborn")
    else:
        evaluate_on_test_set()
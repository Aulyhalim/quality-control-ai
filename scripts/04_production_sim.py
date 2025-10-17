# scripts/04_production_sim.py

import requests
import os
import glob
import shutil
import time

# --- Konfigurasi ---
API_URL = "http://127.0.0.1:8000/inspect/"
INPUT_DIR = "input_for_sim"
OUTPUT_DIR = "output"
APPROVED_DIR = os.path.join(OUTPUT_DIR, "approved")
REJECTED_DIR = os.path.join(OUTPUT_DIR, "rejected")

def run_simulation():
    """
    Mensimulasikan lini produksi: mengambil gambar dari folder input,
    mengirimkannya ke API, dan memindahkannya berdasarkan hasil prediksi.
    """
    print("--- Memulai Simulasi Lini Produksi ---")

    os.makedirs(APPROVED_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)

    image_paths = glob.glob(os.path.join(INPUT_DIR, '*.jpeg')) + \
                  glob.glob(os.path.join(INPUT_DIR, '*.jpg')) + \
                  glob.glob(os.path.join(INPUT_DIR, '*.png'))
    
    total_images = len(image_paths)
    if not image_paths:
        print(f"\n[ERROR] Tidak ada gambar yang ditemukan di folder '{INPUT_DIR}'.")
        print("Silakan salin beberapa gambar ke folder tersebut untuk memulai simulasi.")
        return

    print(f"Ditemukan {total_images} gambar untuk diinspeksi.")

    # --- BARU: Variabel untuk menghitung hasil ---
    approved_count = 0
    rejected_count = 0
    error_count = 0
    start_time = time.time()

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        print(f"\nInspeksi gambar: {filename}")

        try:
            with open(image_path, 'rb') as f:
                files = {'file': (filename, f, 'image/jpeg')}
                response = requests.post(API_URL, files=files, timeout=10)

            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction")
                
                print(f"  -> Hasil API: Prediksi = {prediction}")

                if prediction == "good":
                    shutil.move(image_path, os.path.join(APPROVED_DIR, filename))
                    print(f"  -> Aksi: File dipindahkan ke '{APPROVED_DIR}'")
                    approved_count += 1 # <-- BARU
                elif prediction == "defective":
                    shutil.move(image_path, os.path.join(REJECTED_DIR, filename))
                    print(f"  -> Aksi: File dipindahkan ke '{REJECTED_DIR}'")
                    rejected_count += 1 # <-- BARU
                else:
                    print("  -> Aksi: Prediksi tidak dikenali, file tidak dipindahkan.")
                    error_count += 1 # <-- BARU

            else:
                print(f"  -> Error: API mengembalikan status code {response.status_code}")
                error_count += 1 # <-- BARU

        except requests.exceptions.RequestException as e:
            print(f"  -> Error: Tidak dapat terhubung ke API di {API_URL}. Pastikan server API berjalan.")
            error_count += 1 # <-- BARU
            break

    # --- SELURUH BLOK INI BARU: Tampilkan Ringkasan Hasil ---
    print("\n--- Simulasi Selesai ---")
    
    total_time = time.time() - start_time
    processed_images = approved_count + rejected_count + error_count
    
    print("\n================== RINGKASAN SIMULASI ==================")
    print(f" Waktu Total         : {total_time:.2f} detik")
    print(f" Total Gambar          : {total_images}")
    print(f" Gambar Diproses       : {processed_images}")
    print("----------------------------------------------------")
    print(f" ✅ Diterima (Approved)  : {approved_count}")
    print(f" ❌ Ditolak (Rejected)    : {rejected_count}")
    print(f" 🔥 Gagal Diproses       : {error_count}")
    print("====================================================")
    
    if total_images > 0:
        approved_rate = (approved_count / total_images) * 100
        print(f"\n Tingkat Penerimaan: {approved_rate:.2f}%")

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
    
    run_simulation()
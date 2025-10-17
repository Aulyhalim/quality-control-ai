# scripts/03_quantize_benchmark.py

import torch
from torchvision import models
import os
import time
import matplotlib.pyplot as plt

# --- 1. Konfigurasi ---
# Path ke model yang sudah Anda latih
ORIGINAL_MODEL_PATH = 'models/quality_control_model.pth'
QUANTIZED_MODEL_PATH = 'models/quality_control_model_quantized.ptq'
BENCHMARK_CHART_PATH = 'output/metrics/benchmark_results.png' # <-- BARU: Path untuk menyimpan grafik
NUM_CLASSES = 2 # defective, good

# --- 2. Muat Model Asli (FP32) ---
print("Memuat model asli (FP32)...")

# Kita perlu membuat ulang arsitektur model terlebih dahulu
model_fp32 = models.resnet18()
num_ftrs = model_fp32.fc.in_features
model_fp32.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)

# Muat state_dict (weights) yang sudah terlatih
# Pastikan model dimuat ke CPU untuk konsistensi benchmark
model_fp32.load_state_dict(torch.load(ORIGINAL_MODEL_PATH, map_location=torch.device('cpu')))
model_fp32.eval() # Set ke mode evaluasi

# --- 3. Lakukan Kuantisasi Dinamis ---
print("\nMelakukan kuantisasi dinamis pada model...")
model_quantized = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# --- 4. Simpan Model Terkuantisasi ---
print("Menyimpan model terkuantisasi...")
scripted_quantized_model = torch.jit.script(model_quantized)
scripted_quantized_model.save(QUANTIZED_MODEL_PATH)
print(f"Model terkuantisasi disimpan di: {QUANTIZED_MODEL_PATH}")

# --- 5. Benchmark Performa ---
print("\n--- MEMULAI BENCHMARK ---")

# A. Benchmark Ukuran File
original_size = os.path.getsize(ORIGINAL_MODEL_PATH) / (1024 * 1024)
quantized_size = os.path.getsize(QUANTIZED_MODEL_PATH) / (1024 * 1024)

print(f"\n[Ukuran Model]")
print(f"  - Ukuran Asli (FP32): {original_size:.2f} MB")
print(f"  - Ukuran Terkuantisasi (INT8): {quantized_size:.2f} MB")
print(f"  - Pengurangan Ukuran: {((original_size - quantized_size) / original_size) * 100:.2f}%")

# B. Benchmark Latensi Inferensi
print("\n[Latensi Inferensi (rata-rata dari 100 run)]")
dummy_input = torch.randn(1, 3, 224, 244)

for _ in range(10): # Warm-up
    _ = model_fp32(dummy_input)
    _ = model_quantized(dummy_input)

start_time = time.time()
for _ in range(100): _ = model_fp32(dummy_input)
latency_fp32 = (time.time() - start_time) / 100 * 1000
print(f"  - Latensi Asli (FP32): {latency_fp32:.2f} ms")

start_time = time.time()
for _ in range(100): _ = model_quantized(dummy_input)
latency_int8 = (time.time() - start_time) / 100 * 1000
print(f"  - Latensi Terkuantisasi (INT8): {latency_int8:.2f} ms")
print(f"  - Peningkatan Kecepatan: {(latency_fp32 / latency_int8):.2f}x lebih cepat")

print("\n--- BENCHMARK SELESAI ---")


# --- 6. BUAT DAN SIMPAN GRAFIK (SELURUH BLOK INI BARU) ---
print("\nMembuat grafik benchmark...")
os.makedirs(os.path.dirname(BENCHMARK_CHART_PATH), exist_ok=True)

# Data untuk plot
labels = ['FP32 (Original)', 'INT8 (Quantized)']
sizes = [original_size, quantized_size]
latencies = [latency_fp32, latency_int8]

# Buat plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Perbandingan Kinerja Model: FP32 vs INT8', fontsize=16)

# Grafik Ukuran Model
bars1 = ax1.bar(labels, sizes, color=['skyblue', 'lightgreen'])
ax1.set_ylabel('Ukuran (MB)')
ax1.set_title('Perbandingan Ukuran Model')
ax1.bar_label(bars1, fmt='%.2f MB')

# Grafik Latensi Inferensi
bars2 = ax2.bar(labels, latencies, color=['skyblue', 'lightgreen'])
ax2.set_ylabel('Latensi (ms)')
ax2.set_title('Perbandingan Latensi Inferensi (CPU)')
ax2.bar_label(bars2, fmt='%.2f ms')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(BENCHMARK_CHART_PATH)

print(f"Grafik benchmark disimpan di: {BENCHMARK_CHART_PATH}")
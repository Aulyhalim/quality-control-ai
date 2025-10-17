# scripts/02_train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os
import yaml
from sklearn.metrics import classification_report, accuracy_score
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch

def load_config(config_path='config.yaml'):
    """Memuat file konfigurasi YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model():
    """Fungsi utama untuk melatih, mengevaluasi, dan melacak model."""
    # 1. Muat Konfigurasi
    config = load_config()
    DATA_DIR = config['data_dir']
    MODEL_SAVE_PATH = os.path.join(config['model_dir'], 'quality_control_model.pth')
    OUTPUT_DIR = os.path.join(config['output_dir'], 'metrics')
    NUM_EPOCHS = config['training']['num_epochs']
    BATCH_SIZE = config['training']['batch_size']
    LEARNING_RATE = config['training']['learning_rate']

    # 2. Mulai Sesi Pelacakan MLflow
    mlflow.set_experiment("Quality Control Training")
    with mlflow.start_run() as run:
        print(f"Memulai MLflow Run ID: {run.info.run_id}")
        
        mlflow.log_params({
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'model_architecture': config['model']['name']
        })

        # 3. Definisikan Transformasi Data
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        # 4. Muat Dataset
        print("Memuat dataset...")
        image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) for x in ['train', 'val']}
        dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        
        mlflow.log_param('class_names', class_names)
        mlflow.log_param('training_size', dataset_sizes['train'])
        mlflow.log_param('validation_size', dataset_sizes['val'])
        
        print(f"Kelas yang ditemukan: {class_names}")
        print(f"Ukuran dataset -> Train: {dataset_sizes['train']}, Validation: {dataset_sizes['val']}")

        # 5. Inisialisasi Model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Menggunakan perangkat: {device}")

        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 6. Loop Pelatihan dan Validasi
        print("\nMemulai pelatihan...")
        start_time = time.time()
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(NUM_EPOCHS):
            print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
            print('-' * 15)

            for phase in ['train', 'val']:
                model.train() if phase == 'train' else model.eval()
                running_loss, all_labels, all_preds = 0.0, [], []
                
                progress_bar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}")
                
                for inputs, labels in progress_bar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    
                    # --- PERBAIKAN DI SINI ---
                    # Menampilkan loss dari batch saat ini, bukan rata-rata. Ini lebih aman dan informatif.
                    progress_bar.set_postfix(loss=f'{loss.item():.4f}')

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = accuracy_score(all_labels, all_preds)
                
                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_acc'].append(epoch_acc)
                
                mlflow.log_metric(f'{phase}_loss', epoch_loss, step=epoch)
                mlflow.log_metric(f'{phase}_acc', epoch_acc, step=epoch)
                
                print(f"--> {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                
                if phase == 'val':
                    print("\nLaporan Klasifikasi Validasi:")
                    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

        total_time = time.time() - start_time
        print(f"\nPelatihan selesai dalam {total_time // 60:.0f} menit {total_time % 60:.0f} detik.")
        mlflow.log_metric('training_duration_seconds', total_time)

        # 7. Simpan Model
        print("Menyimpan model...")
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model disimpan di: {MODEL_SAVE_PATH}")

        mlflow.pytorch.log_model(model, "model")
        
        # 8. Buat dan Simpan Grafik
        print("\nMembuat grafik performa...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss'); plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training vs Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy'); plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Training vs Validation Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, 'training_performance.png')
        plt.savefig(plot_path)
        print(f"Grafik disimpan di: {plot_path}")

        mlflow.log_artifact(plot_path)

if __name__ == '__main__':
    train_model()
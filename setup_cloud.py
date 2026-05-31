import os
import zipfile
import shutil
import glob
import re
from huggingface_hub import hf_hub_download

def setup_cloud():
    print("=== Memulai Setup Cloud Vast.ai ===")
    
    # 1. Pastikan struktur direktori ada
    os.makedirs("create-dataset/id", exist_ok=True)
    os.makedirs("StyleTTS2/Models/styletts2", exist_ok=True)

    # 2. Download Dataset dari HuggingFace (repo tipe dataset)
    print("\n[1/4] Mengunduh final_dataset.zip dari eldntr/final_dataset...")
    dataset_zip = hf_hub_download(repo_id="eldntr/final_dataset", filename="final_dataset.zip", repo_type="dataset")
    
    # 3. Ekstrak Dataset
    print("\n[2/4] Mengekstrak dataset ke create-dataset/id/...")
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall("create-dataset/id/")
    print("Dataset berhasil diekstrak!")

    # 4. Download Pretrained Weights StyleTTS2
    print("\n[3/4] Mengunduh bobot pretrained StyleTTS2-LJSpeech...")
    weights_path = hf_hub_download(repo_id="yl4579/StyleTTS2-LJSpeech", filename="Models/LJSpeech/epoch_2nd_00100.pth")
    
    # Salin bobot ke root StyleTTS2 agar mudah diakses
    shutil.copy(weights_path, "StyleTTS2/epoch_2nd_00100.pth")
    print("Bobot berhasil disalin ke StyleTTS2/epoch_2nd_00100.pth")

    # 5. Otomatisasi pembaruan Configs
    print("\n[4/4] Memperbarui seluruh file konfigurasi (.yml)...")
    configs_dir = "StyleTTS2/Configs"
    yml_files = glob.glob(f"{configs_dir}/**/*.yml", recursive=True)

    for file_path in yml_files:
        with open(file_path, "r") as f:
            content = f.read()

        # Update jalur ASR
        content = re.sub(r'ASR_config:\s*".*?"', 'ASR_config: "Utils/ASR/config.yml"', content)
        content = re.sub(r'ASR_path:\s*".*?"', 'ASR_path: "Utils/ASR/epoch_00200.pth"', content)

        # Update jalur Data
        content = re.sub(r'train_data:\s*".*?"', 'train_data: "../create-dataset/id/final_dataset/phonemized_lists/train_list_phon.txt"', content)
        content = re.sub(r'val_data:\s*".*?"', 'val_data: "../create-dataset/id/final_dataset/phonemized_lists/val_list_phon.txt"', content)
        content = re.sub(r'root_path:\s*".*?"', 'root_path: "../create-dataset/id/final_dataset/wavs"', content)
        content = re.sub(r'OOD_data:\s*".*?"', 'OOD_data: "../create-dataset/id/final_dataset/phonemized_lists/test_list_phon.txt"', content)

        # Update jalur pretrained model ke file yang baru didownload
        # Jika file config_second.yml atau config_ft.yml memakai epoch_2nd_...
        content = re.sub(r'pretrained_model:\s*".*epoch_2nd_.*?"', 'pretrained_model: "epoch_2nd_00100.pth"', content)

        with open(file_path, "w") as f:
            f.write(content)
            
    print(f"Berhasil memperbarui {len(yml_files)} file konfigurasi.")
    print("\n=== Setup Selesai! Anda siap untuk menjalankan training di Vast.ai ===")

if __name__ == "__main__":
    setup_cloud()

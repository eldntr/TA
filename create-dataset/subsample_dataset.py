import os
import glob
import random
import argparse

def generate_subsampled_files(lang_dir, ratio=0.3, seed=42):
    lang_name = os.path.basename(lang_dir)
    phon_dir = os.path.join(lang_dir, "final_dataset", "phonemized_lists")
    
    if not os.path.exists(phon_dir):
        print(f"Error: Folder {phon_dir} tidak ditemukan.")
        return
        
    print(f"\n--- Membuat File Subsample untuk [{lang_name.upper()}] (Rasio: {ratio*100:.0f}%) ---")
    
    # Cari semua file list fonem asli (*_phon.txt) tapi bukan yang (*_phon_sub.txt)
    original_lists = glob.glob(os.path.join(phon_dir, "*_list_phon.txt"))
    
    for filepath in original_lists:
        filename = os.path.basename(filepath)
        sub_filename = filename.replace(".txt", "_sub.txt")
        sub_filepath = os.path.join(phon_dir, sub_filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        random.seed(seed)
        total_lines = len(lines)
        sample_size = max(1, int(total_lines * ratio))
        sampled_lines = random.sample(lines, sample_size)
        sampled_lines.sort() # urutkan kembali biar rapi
        
        with open(sub_filepath, 'w', encoding='utf-8') as f:
            f.writelines(sampled_lines)
            
        print(f"  - Generated: {sub_filename} ({total_lines} -> {len(sampled_lines)} baris)")

def update_configs(configs_dir, use_sub=True):
    # Cari semua file .yml di folder Configs
    config_files = []
    for root, _, files in os.walk(configs_dir):
        for file in files:
            if file.endswith('.yml') or file.endswith('.yaml'):
                config_files.append(os.path.join(root, file))
                
    print(f"\n--- Mengubah Konfigurasi di {configs_dir} ---")
    
    replaced_count = 0
    for filepath in config_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        new_content = content
        if use_sub:
            # Ganti dari full ke subsampled
            # Hindari penggantian berulang jika sudah menggunakan _sub.txt
            if "train_list_phon_sub.txt" not in new_content:
                new_content = new_content.replace("train_list_phon.txt", "train_list_phon_sub.txt")
            if "val_list_phon_sub.txt" not in new_content:
                new_content = new_content.replace("val_list_phon.txt", "val_list_phon_sub.txt")
            if "test_list_phon_sub.txt" not in new_content:
                new_content = new_content.replace("test_list_phon.txt", "test_list_phon_sub.txt")
        else:
            # Kembalikan ke full
            new_content = new_content.replace("train_list_phon_sub.txt", "train_list_phon.txt")
            new_content = new_content.replace("val_list_phon_sub.txt", "val_list_phon.txt")
            new_content = new_content.replace("test_list_phon_sub.txt", "test_list_phon.txt")
            
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  - Diperbarui: {os.path.relpath(filepath, configs_dir)}")
            replaced_count += 1
            
    print(f"Total {replaced_count} file konfigurasi diperbarui.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample StyleTTS2 lists and toggle config files.")
    parser.add_argument("--ratio", type=float, default=0.3, help="Rasio data yang akan digunakan (default: 0.3 untuk 30%)")
    parser.add_argument("--restore", action="store_true", help="Kembalikan file konfigurasi ke 100% full data")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    configs_dir = os.path.join(project_root, "StyleTTS2", "Configs")
    
    if args.restore:
        print("==================================================")
        print("Menghubungkan Ulang Konfigurasi ke 100% Full Data")
        print("==================================================")
        update_configs(configs_dir, use_sub=False)
    else:
        print("==================================================")
        print(f"Membuat File Subsample 30% & Update Konfigurasi")
        print("==================================================")
        
        # 1. Generate file list 30% terpisah (*_phon_sub.txt)
        for lang in ["id", "jv"]:
            lang_path = os.path.join(script_dir, lang)
            if os.path.exists(lang_path):
                generate_subsampled_files(lang_path, ratio=args.ratio)
                
        # 2. Update config files untuk menunjuk ke file *_phon_sub.txt tersebut
        update_configs(configs_dir, use_sub=True)
        
    print("\nSelesai!")

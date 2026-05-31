import os
import sys
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm

# Menambahkan root folder TA ke sys.path agar bisa melakukan import AuxiliaryASR
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from AuxiliaryASR.custom_phonemizer import Phonemizer

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    final_dataset_dir = os.path.join(script_dir, "final_dataset")
    wavs_dir = os.path.join(final_dataset_dir, "wavs")
    
    # Membuat folder tersendiri agar rapi di dalam final_dataset
    phon_dir = os.path.join(final_dataset_dir, "phonemized_lists")
    os.makedirs(phon_dir, exist_ok=True)
    
    print("Menginisialisasi Phonemizer (Lingua Language Detector & Espeak)...")
    phonemizer = Phonemizer()
    
    splits_dir = os.path.join(final_dataset_dir, "list")
    splits = ["train_list.txt", "val_list.txt", "test_list.txt"]
    
    valid_durations = []
    total_duration_sec = 0.0
    all_metadata_lines = []
    
    removed_count = 0
    
    for split_file in splits:
        input_path = os.path.join(splits_dir, split_file)
        
        # Penamaan output file, contoh: train_list_phon.txt
        out_name = split_file.replace(".txt", "_phon.txt")
        output_phon_path = os.path.join(phon_dir, out_name)
        
        if not os.path.exists(input_path):
            continue
            
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        clean_split_lines = []
        phon_lines = []
        
        print(f"\nMemproses {split_file}...")
        for line in tqdm(lines, desc="Phonemizing & Checking"):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('|')
            if len(parts) == 3:
                filename, transcript, speaker_id = parts
            elif len(parts) == 2:
                filename, transcript = parts
                speaker_id = ""
            else:
                print(f"Format tidak dikenali, melewati: {line}")
                continue
                
            # Proses fonemisasi kata per kata
            words = transcript.split()
            phonemes = []
            for word in words:
                phonemes.append(phonemizer(word))
                
            phonemized_transcript = " ".join(phonemes)
            wav_path = os.path.join(wavs_dir, filename)
            
            # Cek panjang karakter fonem
            if len(phonemized_transcript) > 512:
                # Hapus file wav jika panjangnya > 512
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                removed_count += 1
                continue
                
            # Jika valid, ambil durasinya dan catat ke file clean
            if os.path.exists(wav_path):
                info = torchaudio.info(wav_path)
                duration = info.num_frames / info.sample_rate
                valid_durations.append(duration)
                total_duration_sec += duration
                
                clean_split_lines.append(line + "\n")
                all_metadata_lines.append(f"{filename}|{transcript}\n")
                
                if speaker_id:
                    phon_lines.append(f"{filename}|{phonemized_transcript}|{speaker_id}\n")
                else:
                    phon_lines.append(f"{filename}|{phonemized_transcript}\n")
                    
        # Perbarui file split list asli (buang yang > 512)
        with open(input_path, 'w', encoding='utf-8') as f:
            f.writelines(clean_split_lines)
            
        # Simpan file phonemized list
        with open(output_phon_path, 'w', encoding='utf-8') as f:
            f.writelines(phon_lines)
            
    # Perbarui metadata.csv
    metadata_path = os.path.join(final_dataset_dir, "metadata.csv")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.writelines(all_metadata_lines)
        
    # Perbarui total_duration.txt
    hours = int(total_duration_sec // 3600)
    minutes = int((total_duration_sec % 3600) // 60)
    seconds = total_duration_sec % 60
    
    out_duration_txt = os.path.join(final_dataset_dir, "total_duration.txt")
    duration_text = f"Total Durasi Dataset Final: {hours} jam {minutes} menit {seconds:.2f} detik\nTotal detik: {total_duration_sec:.2f}\n"
    with open(out_duration_txt, "w", encoding="utf-8") as f:
        f.write(duration_text)
        
    # Perbarui distribusi.png
    print("\nMemperbarui grafik distribusi durasi...")
    plt.figure(figsize=(10, 6))
    plt.hist(valid_durations, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribusi Durasi Audio (Final Dataset - Post Phonemization)')
    plt.xlabel('Durasi (detik)')
    plt.ylabel('Jumlah File')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    graph_path = os.path.join(final_dataset_dir, "duration_distribution.png")
    plt.savefig(graph_path)
        
    print("\n--- Semua proses fonemisasi dan pembersihan selesai! ---")
    print(f"Total file dihapus (> 512 char): {removed_count}")
    print(f"Total file valid tersisa     : {len(valid_durations)}")
    print(f"Total Durasi Baru            : {hours} jam {minutes} menit {seconds:.2f} detik")
    print(f"Hasil list fonem disimpan di : {phon_dir}")

if __name__ == "__main__":
    main()

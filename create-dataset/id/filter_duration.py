import os
import shutil
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm

def filter_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    transcribe_dir = os.path.join(script_dir, "transcribe")
    
    # Lokasi file sumber
    metadata_path = os.path.join(transcribe_dir, "processed_dataset", "metadata.csv")
    wavs_original_dir = os.path.join(transcribe_dir, "processed_dataset", "wavs")
    wavs_restored_dir = os.path.join(transcribe_dir, "processed_dataset_restored", "wavs")
    
    # Lokasi file tujuan (dataset bersih dan difilter)
    output_dir = os.path.join(script_dir, "final_dataset")
    out_wavs = os.path.join(output_dir, "wavs")
    out_metadata = os.path.join(output_dir, "metadata.csv")
    
    os.makedirs(out_wavs, exist_ok=True)
    
    print(f"Membaca metadata dari: {metadata_path}")
    if not os.path.exists(metadata_path):
        print("Error: metadata.csv tidak ditemukan!")
        return

    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    valid_lines = []
    valid_durations = []
    total_duration_sec = 0.0
    
    print("Menganalisis dan menyalin file audio (Durasi: 2 - 11 detik)...")
    
    for line in tqdm(lines):
        line = line.strip()
        if not line: 
            continue
            
        parts = line.split('|')
        filename = parts[0]
        
        orig_path = os.path.join(wavs_original_dir, filename)
        restored_path = os.path.join(wavs_restored_dir, filename)
        
        # Pastikan file wav restored ada
        if not os.path.exists(restored_path):
            continue
            
        try:
            # Mengambil metadata file audio dengan efisien
            info = torchaudio.info(restored_path)
            duration = info.num_frames / info.sample_rate
            
            # Filter durasi 2 sampai 11 detik
            if 2.0 <= duration <= 11.0:
                valid_lines.append(line + "\n")
                valid_durations.append(duration)
                total_duration_sec += duration
                
                # Hanya copy file yang sudah direstorasi ke dalam folder wavs
                shutil.copy2(restored_path, os.path.join(out_wavs, filename))
                    
        except Exception as e:
            print(f"Gagal memproses {filename}: {e}")
            
    # Menyimpan metadata baru yang hanya berisi file 2-11 detik
    with open(out_metadata, 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)
        
    hours = int(total_duration_sec // 3600)
    minutes = int((total_duration_sec % 3600) // 60)
    seconds = total_duration_sec % 60
    
    out_duration_txt = os.path.join(output_dir, "total_duration.txt")
    duration_text = f"Total Durasi Dataset Final: {hours} jam {minutes} menit {seconds:.2f} detik\nTotal detik: {total_duration_sec:.2f}\n"
    with open(out_duration_txt, "w", encoding="utf-8") as f:
        f.write(duration_text)
        
    print("\nMembuat grafik distribusi durasi...")
    plt.figure(figsize=(10, 6))
    plt.hist(valid_durations, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribusi Durasi Audio (Final Dataset)')
    plt.xlabel('Durasi (detik)')
    plt.ylabel('Jumlah File')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    graph_path = os.path.join(output_dir, "duration_distribution.png")
    plt.savefig(graph_path)
        
    print("\n--- Proses Selesai ---")
    print(f"Total awal   : {len(lines)} file")
    print(f"Total valid  : {len(valid_lines)} file (durasi 2-11s)")
    print(f"File dibuang : {len(lines) - len(valid_lines)} file")
    print(f"Total Durasi Dataset Final: {hours} jam {minutes} menit {seconds:.2f} detik")
    print(f"Dataset final tersimpan di: {output_dir}")
    print(f"Grafik distribusi disimpan di: {graph_path}")

if __name__ == "__main__":
    filter_dataset()

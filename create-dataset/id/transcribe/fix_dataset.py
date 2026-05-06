import os
import glob

def fix_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, "processed_dataset")
    metadata_path = os.path.join(processed_dir, "metadata.csv")
    speaker_map_path = os.path.join(processed_dir, "speaker_map.txt")
    wavs_dir = os.path.join(processed_dir, "wavs")
    
    # 1. Tambahkan speaker baru ke speaker_map.txt
    with open(speaker_map_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Cek apakah 018 sudah ada untuk mencegah duplikasi
    has_018 = any("018" in line for line in lines)
    if not has_018:
        with open(speaker_map_path, "a", encoding="utf-8") as f:
            f.write("Layar Kertas 2|018\n")
            print("Berhasil menambahkan 'Layar Kertas 2|018' ke speaker_map.txt")
    else:
        print("Speaker 018 sudah ada di speaker_map.txt")
    
    # 2. Update metadata.csv
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_lines = f.readlines()
        
    new_meta_lines = []
    updated_count = 0
    
    # Video ID yang akan diubah: 203 sampai 213
    vids_to_change = [f"{i:03d}" for i in range(203, 214)]
    
    for line in meta_lines:
        # Format baris: 015_203_0001.wav|teks transcript
        parts = line.split("|", 1)
        if len(parts) == 2:
            filename = parts[0]
            file_parts = filename.split("_")
            # Cek jika file milik speaker 015 (Layar Kertas) dan termasuk video target
            if len(file_parts) == 3 and file_parts[0] == "015":
                vid = file_parts[1]
                if vid in vids_to_change:
                    new_filename = f"018_{vid}_{file_parts[2]}"
                    line = f"{new_filename}|{parts[1]}"
                    updated_count += 1
        new_meta_lines.append(line)
        
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.writelines(new_meta_lines)
    print(f"Berhasil mengupdate {updated_count} baris di metadata.csv")
    
    # 3. Rename file-file di folder wavs
    renamed_count = 0
    for vid in vids_to_change:
        pattern = os.path.join(wavs_dir, f"015_{vid}_*.wav")
        files_to_rename = glob.glob(pattern)
        for old_path in files_to_rename:
            basename = os.path.basename(old_path)
            # Ubah "015" di awal menjadi "018"
            new_basename = "018" + basename[3:]
            new_path = os.path.join(wavs_dir, new_basename)
            os.rename(old_path, new_path)
            renamed_count += 1
            
    print(f"Berhasil merename {renamed_count} file .wav di direktori wavs")

if __name__ == "__main__":
    fix_dataset()

import os
import random

def create_splits(train_ratio=0.9, val_ratio=0.05, test_ratio=0.05):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    id_dir = os.path.dirname(script_dir)
    processed_dir = os.path.join(id_dir, "final_dataset")
    metadata_path = os.path.join(processed_dir, "metadata.csv")
    
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found at {metadata_path}")
        return
        
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Expected format in metadata.csv: filename|transcript
        parts = line.split('|', 1)
        if len(parts) != 2:
            continue
            
        filename = parts[0]
        transcript = parts[1]
        
        # Ekstrak ID speaker dari awalan filename: misal "015_203_0001.wav" -> "015"
        speaker_id_str = filename.split('_')[0]
        try:
            # Convert to int to remove leading zeros as requested (e.g., "015" -> "15")
            speaker_id = str(int(speaker_id_str))
        except ValueError:
            speaker_id = speaker_id_str
            
        # Format akhir: filename|transcript|speaker_id
        new_line = f"{filename}|{transcript}|{speaker_id}\n"
        formatted_lines.append(new_line)
        
    # Shuffle the dataset
    random.seed(42) # Seed untuk memastikan split yang konsisten jika dijalankan ulang
    random.shuffle(formatted_lines)
    
    total = len(formatted_lines)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_lines = formatted_lines[:train_end]
    val_lines = formatted_lines[train_end:val_end]
    test_lines = formatted_lines[val_end:]
    
    # Path untuk menyimpan list output
    train_path = os.path.join(processed_dir, "train_list.txt")
    val_path = os.path.join(processed_dir, "val_list.txt")
    test_path = os.path.join(processed_dir, "test_list.txt")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(val_path, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
        
    print(f"Berhasil membuat split data di {processed_dir}:")
    print(f"Total baris : {total}")
    print(f"Train list  : {len(train_lines)} file")
    print(f"Val list    : {len(val_lines)} file")
    print(f"Test list   : {len(test_lines)} file")

if __name__ == "__main__":
    create_splits()

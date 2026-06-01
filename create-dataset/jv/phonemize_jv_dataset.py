import os
import re
import sys
import shutil
import random
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
from phonemizer.backend import EspeakBackend

# Add root folder of the project to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

class JavanesePhonemizer:
    def __init__(self):
        # Match the Indonesian phonemization baseline used in
        # AuxiliaryASR/custom_phonemizer.py.
        self.backend = EspeakBackend(
            language='id',
            preserve_punctuation=True,
            with_stress=True
        )

    def clean_text(self, text: str) -> str:
        text = text.lower().strip()

        # Strip dataset tags like _letter, _prep, _conj, _pron, _part, _adv,
        # _noun, _adj without rewriting the token into a spoken letter name.
        text = re.sub(r'_letter(?:-en)?', '', text)
        text = re.sub(r'_(prep|conj|pron|part|adv|noun|adj|verb)', '', text)

        # Clean up extra hyphens or symbols
        text = text.replace('-', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def __call__(self, text: str) -> str:
        cleaned = self.clean_text(text)
        return self.backend.phonemize([cleaned], strip=True)[0]

def string_punctuation_custom():
    return ';:,.!?¡¿—…"«»“”()'

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dataset_dir = os.path.join(script_dir, "jv_id_male")
    source_wavs_dir = os.path.join(source_dataset_dir, "wavs")
    line_index_path = os.path.join(source_dataset_dir, "line_index.tsv")
    
    final_dataset_dir = os.path.join(script_dir, "final_dataset")
    final_wavs_dir = os.path.join(final_dataset_dir, "wavs")
    phon_dir = os.path.join(final_dataset_dir, "phonemized_lists")
    
    os.makedirs(final_wavs_dir, exist_ok=True)
    os.makedirs(phon_dir, exist_ok=True)
    
    if not os.path.exists(line_index_path):
        print(f"Error: line_index.tsv tidak ditemukan di {line_index_path}")
        return
        
    print("Menginisialisasi Javanese Phonemizer...")
    phonemizer = JavanesePhonemizer()
    
    print("Membaca line_index.tsv...")
    with open(line_index_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    valid_entries = []
    removed_count = 0
    total_duration_sec = 0.0
    valid_durations = []
    
    print("Memproses dan memfonemisasi dataset...")
    for line in tqdm(lines, desc="Phonemizing"):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('\t')
        if len(parts) < 3:
            continue
            
        audio_id, _, transcript = parts[:3]
        filename = f"{audio_id}.wav"
        src_wav_path = os.path.join(source_wavs_dir, filename)
        dest_wav_path = os.path.join(final_wavs_dir, filename)
        
        if not os.path.exists(src_wav_path):
            continue
            
        # Get speaker ID from audio_id (e.g. jvm_00027_xxx -> 27)
        audio_parts = audio_id.split('_')
        if len(audio_parts) >= 2:
            try:
                speaker_id = str(int(audio_parts[1]))
            except ValueError:
                speaker_id = audio_parts[1]
        else:
            speaker_id = "0"
            
        # Phonemize the transcript
        phonemized_transcript = phonemizer(transcript)
        
        # Check phoneme token length limit
        phoneme_tokens = phonemized_transcript.split()
        if len(phoneme_tokens) > 512:
            removed_count += 1
            continue
            
        # Copy audio file to final_dataset
        shutil.copy(src_wav_path, dest_wav_path)
        
        # Get audio duration
        info = torchaudio.info(dest_wav_path)
        duration = info.num_frames / info.sample_rate
        valid_durations.append(duration)
        total_duration_sec += duration
        
        # Record valid entry: format = filename|phonemes|speaker_id
        valid_entries.append({
            'filename': filename,
            'transcript': transcript,
            'phonemes': phonemized_transcript,
            'speaker_id': speaker_id
        })
        
    print(f"\nTotal entri valid: {len(valid_entries)}")
    print(f"Total entri dibuang (> 512 token fonem): {removed_count}")
    
    # Map speaker IDs starting from 0
    unique_speakers = sorted(list(set(item['speaker_id'] for item in valid_entries)), key=lambda x: int(x) if x.isdigit() else x)
    spk_to_idx = {spk_id: str(idx) for idx, spk_id in enumerate(unique_speakers)}
    
    # Save speaker mapping to a file
    mapping_path = os.path.join(final_dataset_dir, "speaker_mapping.txt")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        for spk_id, idx in spk_to_idx.items():
            f.write(f"{spk_id}|{idx}\n")
    print(f"File mapping speaker disimpan di: {mapping_path}")
    
    # Update speaker_id to the mapped contiguous integer in valid_entries
    for item in valid_entries:
        item['speaker_id'] = spk_to_idx[item['speaker_id']]
    
    # Shuffle and split dataset (90% train, 5% val, 5% test)
    random.seed(42)
    random.shuffle(valid_entries)
    
    total = len(valid_entries)
    train_end = int(total * 0.9)
    val_end = train_end + int(total * 0.05)
    
    train_data = valid_entries[:train_end]
    val_data = valid_entries[train_end:val_end]
    test_data = valid_entries[val_end:]
    
    # Write lists
    splits = {
        "train_list_phon.txt": train_data,
        "val_list_phon.txt": val_data,
        "test_list_phon.txt": test_data
    }
    
    for filename, data in splits.items():
        filepath = os.path.join(phon_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(f"{item['filename']}|{item['phonemes']}|{item['speaker_id']}\n")
                
    # Save metadata.csv
    metadata_path = os.path.join(final_dataset_dir, "metadata.csv")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for item in valid_entries:
            f.write(f"{item['filename']}|{item['transcript']}\n")
            
    # Save total_duration.txt
    hours = int(total_duration_sec // 3600)
    minutes = int((total_duration_sec % 3600) // 60)
    seconds = total_duration_sec % 60
    
    out_duration_txt = os.path.join(final_dataset_dir, "total_duration.txt")
    duration_text = f"Total Durasi Dataset Final Jawa: {hours} jam {minutes} menit {seconds:.2f} detik\nTotal detik: {total_duration_sec:.2f}\n"
    with open(out_duration_txt, "w", encoding="utf-8") as f:
        f.write(duration_text)
        
    # Plot duration distribution
    print("\nMembuat grafik distribusi durasi...")
    plt.figure(figsize=(10, 6))
    plt.hist(valid_durations, bins=50, color='lightgreen', edgecolor='black')
    plt.title('Distribusi Durasi Audio (Javanese Final Dataset)')
    plt.xlabel('Durasi (detik)')
    plt.ylabel('Jumlah File')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    graph_path = os.path.join(final_dataset_dir, "duration_distribution.png")
    plt.savefig(graph_path)
    plt.close()
    
    print("\n--- Proses pembuatan dataset dan fonemisasi selesai! ---")
    print(f"Hasil list fonem disimpan di: {phon_dir}")
    print(f"Total durasi                 : {hours} jam {minutes} menit {seconds:.2f} detik")
    print(f"Distribusi grafik disimpan ke : {graph_path}")

if __name__ == "__main__":
    main()

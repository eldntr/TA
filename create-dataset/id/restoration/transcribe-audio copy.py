import os
import glob
import pandas as pd
import whisperx
import torchaudio
import torch
import gc

def process_audio_files(input_dir, output_dir, metadata_path, speaker_map_path, judul_map_path, duration_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"Mempersiapkan WhisperX di {device} dengan komputasi {compute_type}...")
    
    print("Memuat model WhisperX large-v3...")
    model = whisperx.load_model("large-v3", device, compute_type=compute_type, language="id")

    print("Memuat model alignment untuk bahasa Indonesia...")
    model_a, metadata = whisperx.load_align_model(language_code="id", device=device, model_name="indonesian-nlp/wav2vec2-large-xlsr-indonesian")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    audio_files = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)
    metadata_records = []
    total_duration_sec = 0.0
    
    print(f"Menemukan {len(audio_files)} file audio.")
    
    speaker_dict = {}
    judul_dict = {}
    speaker_counter = 1
    judul_counter = 1
    
    for audio_path in audio_files:
        channel_name = os.path.basename(os.path.dirname(audio_path))
        judul_audio = os.path.splitext(os.path.basename(audio_path))[0]
        
        if channel_name not in speaker_dict:
            speaker_dict[channel_name] = f"{speaker_counter:03d}"
            speaker_counter += 1
            
        if judul_audio not in judul_dict:
            judul_dict[judul_audio] = f"{judul_counter:03d}"
            judul_counter += 1
            
    with open(speaker_map_path, "w", encoding="utf-8") as f:
        for channel, spk in speaker_dict.items():
            f.write(f"{channel}|{spk}\n")
            
    with open(judul_map_path, "w", encoding="utf-8") as f:
        for judul, vid in judul_dict.items():
            f.write(f"{judul}|{vid}\n")
            
    print(f"Berhasil membuat mapping: {len(speaker_dict)} speaker dan {len(judul_dict)} judul.")
    
    for audio_path in audio_files:
        print(f"Memproses: {audio_path}")
        try:
            channel_name = os.path.basename(os.path.dirname(audio_path))
            judul_audio = os.path.splitext(os.path.basename(audio_path))[0]
            spk_code = speaker_dict[channel_name]
            vid_code = judul_dict[judul_audio]
            
            audio = whisperx.load_audio(audio_path)
            
            result = model.transcribe(audio, batch_size=16)
            
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            
            waveform, sample_rate = torchaudio.load(audio_path)
            
            pad_frames = int(0.2 * sample_rate)
            max_frames = waveform.shape[1]
            
            segment_idx = 1
            
            for segment in result["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"].strip()
                
                if not text or (end_time - start_time) < 1.0:
                    continue
                
                start_frame = max(0, int(start_time * sample_rate) - pad_frames)
                end_frame = min(max_frames, int(end_time * sample_rate) + pad_frames)
                
                segment_waveform = waveform[:, start_frame:end_frame]
                
                out_filename = f"{spk_code}_{vid_code}_{segment_idx:04d}.wav"
                out_filepath = os.path.join(output_dir, out_filename)
                torchaudio.save(out_filepath, segment_waveform, sample_rate)
                
                total_duration_sec += (end_frame - start_frame) / sample_rate
                
                metadata_records.append({
                    "audio_path": out_filename,
                    "transcript": text
                })
                
                segment_idx += 1
                
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Gagal memproses {audio_path}: {e}")
            
    df = pd.DataFrame(metadata_records)
    df.to_csv(metadata_path, sep="|", index=False, header=False) 
    print(f"Selesai! Menyimpan {len(metadata_records)} segmen audio ke {metadata_path}")
    
    hours = int(total_duration_sec // 3600)
    minutes = int((total_duration_sec % 3600) // 60)
    seconds = total_duration_sec % 60
    duration_text = f"Total Durasi Dataset: {hours} jam {minutes} menit {seconds:.2f} detik\nTotal detik: {total_duration_sec:.2f}\n"
    with open(duration_path, "w", encoding="utf-8") as f:
        f.write(duration_text)
    print(duration_text)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Default to restored audio if it exists, otherwise warn
    restored_dir = os.path.join(os.path.dirname(script_dir), "dataset_audio_restored")
    raw_dir = os.path.join(os.path.dirname(script_dir), "dataset_audio")
    
    if os.path.exists(restored_dir) and any(os.scandir(restored_dir)):
        input_directory = restored_dir
        print(f"Menggunakan audio yang telah direstorasi dari: {input_directory}")
    else:
        input_directory = raw_dir
        if not os.path.exists(raw_dir):
            print(f"Folder {raw_dir} tidak ditemukan! Pastikan sudah menjalankan download-audio.py.")
        else:
            print(f"WARNING: Folder restorasi '{restored_dir}' tidak ditemukan atau kosong.")
            print(f"Menggunakan audio mentah dari: {input_directory}")
            print("Disarankan menjalankan restoration/restore-audio.py terlebih dahulu untuk kualitas lebih baik.")
    
    processed_dir = os.path.join(script_dir, "processed_dataset")
    output_directory = os.path.join(processed_dir, "wavs")
    metadata_file = os.path.join(processed_dir, "metadata.csv")
    speaker_map_file = os.path.join(processed_dir, "speaker_map.txt")
    judul_map_file = os.path.join(processed_dir, "judul_map.txt")
    duration_file = os.path.join(processed_dir, "total_duration.txt")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(input_directory):
        print(f"Folder {input_directory} tidak ditemukan!")
    else:
        process_audio_files(input_directory, output_directory, metadata_file, speaker_map_file, judul_map_file, duration_file)

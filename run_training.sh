#!/bin/bash

# --- Memeriksa Argumen ---
if [ "$#" -ne 2 ]; then
    echo "Penggunaan: ./run_training.sh <skenario> <tahap>"
    echo ""
    echo "Pilihan <skenario>:"
    echo "  - styletts2"
    echo "  - lpep"
    echo "  - ppim"
    echo "  - lpep_ppim"
    echo ""
    echo "Pilihan <tahap>:"
    echo "  - first  (Menjalankan train_first.py)"
    echo "  - second (Menjalankan train_second.py)"
    echo "  - ft     (Menjalankan train_finetune.py)"
    echo ""
    echo "Contoh: ./run_training.sh lpep second"
    exit 1
fi

SCENARIO=$1
STAGE=$2

# --- Validasi Tahap & Script ---
if [ "$STAGE" == "first" ]; then
    SCRIPT="train_first.py"
elif [ "$STAGE" == "second" ]; then
    SCRIPT="train_second.py"
elif [ "$STAGE" == "ft" ]; then
    SCRIPT="train_finetune.py"
else
    echo "Error: Tahap '$STAGE' tidak valid. Gunakan 'first', 'second', atau 'ft'."
    exit 1
fi

# --- Mendeteksi Jumlah GPU ---
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    echo "Peringatan: nvidia-smi tidak ditemukan. Menggunakan default 1 GPU."
    NUM_GPUS=1
fi

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: Tidak ada GPU yang terdeteksi! Training dibatalkan."
    exit 1
fi

# --- Mendefinisikan Path Konfigurasi ---
CONFIG_PATH="./Configs/${SCENARIO}/config_${STAGE}.yml"

# --- Masuk ke Folder StyleTTS2 ---
cd StyleTTS2 || { echo "Error: Folder StyleTTS2 tidak ditemukan"; exit 1; }

# Cek apakah file konfigurasi ada
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: File konfigurasi tidak ditemukan: StyleTTS2/$CONFIG_PATH"
    exit 1
fi

echo "========================================================="
echo "  Memulai Training StyleTTS2"
echo "  Skenario : $SCENARIO"
echo "  Tahap    : $STAGE ($SCRIPT)"
echo "  Jumlah GPU Terdeteksi: $NUM_GPUS GPU"
echo "  Config   : $CONFIG_PATH"
echo "========================================================="

# --- Menjalankan Accelerate Launch ---
ACCELERATE="../.venv/bin/accelerate"

if [ ! -f "$ACCELERATE" ]; then
    echo "Error: Command accelerate tidak ditemukan di $ACCELERATE"
    echo "Pastikan Anda sudah menginstal dependensi dengan uv sync atau init_vastai.sh"
    exit 1
fi

$ACCELERATE launch \
    --mixed_precision=fp16 \
    --num_processes=$NUM_GPUS \
    $SCRIPT \
    --config_path $CONFIG_PATH

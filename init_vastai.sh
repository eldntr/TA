#!/bin/bash
echo "=== Inisialisasi Environment Vast.ai ==="

# 1. Install uv jika belum ada
if ! command -v uv &> /dev/null; then
    echo "[1/3] Menginstal 'uv' (Ultra-fast Python package installer)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/3] 'uv' sudah terinstal."
fi

# 2. Buat virtual environment dengan uv
if [ ! -d ".venv" ]; then
    echo "[2/3] Membuat Virtual Environment (.venv) dengan uv (Python 3.11)..."
    uv venv --python 3.11
else
    echo "[2/3] Virtual Environment (.venv) sudah ada."
fi

# 3. Install dependensi
echo "[3/3] Menginstal dependensi..."
# Menggunakan uv sync yang otomatis membaca pyproject.toml tanpa error build
uv sync

echo ""
echo "=== Environment siap! Menjalankan setup_cloud.py ==="
# Langsung memanggil python dari dalam venv tanpa perlu source activate
.venv/bin/python setup_cloud.py

echo ""
echo "Untuk mengaktifkan environment di terminal ini nanti, jalankan:"
echo "source .venv/bin/activate"
import glob
import re

configs_dir = "/home/user/TA-Eldin/TA/StyleTTS2/Configs"
yml_files = glob.glob(f"{configs_dir}/**/*.yml", recursive=True)

for file_path in yml_files:
    with open(file_path, "r") as f:
        content = f.read()

    # Update batch_size to 24
    content = re.sub(r'batch_size:\s*\d+', 'batch_size: 24', content)
        
    with open(file_path, "w") as f:
        f.write(content)

print(f"Berhasil mengubah batch_size menjadi 24 pada {len(yml_files)} file konfigurasi.")

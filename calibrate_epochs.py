import glob
import re

configs_dir = "/home/user/TA-Eldin/TA/StyleTTS2/Configs"
yml_files = glob.glob(f"{configs_dir}/**/*.yml", recursive=True)

for file_path in yml_files:
    with open(file_path, "r") as f:
        content = f.read()

    # Untuk config_first.yml
    if "config_first.yml" in file_path:
        content = re.sub(r'TMA_epoch:\s*\d+', 'TMA_epoch: 20', content)
        content = re.sub(r'diff_epoch:\s*\d+', 'diff_epoch: 5', content)
        content = re.sub(r'joint_epoch:\s*\d+', 'joint_epoch: 10', content)
    
    # Untuk config_second.yml
    elif "config_second.yml" in file_path:
        content = re.sub(r'TMA_epoch:\s*\d+', 'TMA_epoch: 15', content)
        content = re.sub(r'diff_epoch:\s*\d+', 'diff_epoch: 5', content)
        content = re.sub(r'joint_epoch:\s*\d+', 'joint_epoch: 10', content)
        
    # Untuk config_ft.yml
    elif "config_ft.yml" in file_path:
        content = re.sub(r'diff_epoch:\s*\d+', 'diff_epoch: 5', content)
        content = re.sub(r'joint_epoch:\s*\d+', 'joint_epoch: 10', content)
        
    with open(file_path, "w") as f:
        f.write(content)

print(f"Berhasil mengkalibrasi epoch parameter pada {len(yml_files)} file konfigurasi.")

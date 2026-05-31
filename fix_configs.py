import os
import glob
import re

configs_dir = "/home/user/TA-Eldin/TA/StyleTTS2/Configs"
yml_files = glob.glob(f"{configs_dir}/**/*.yml", recursive=True)

for file_path in yml_files:
    with open(file_path, "r") as f:
        content = f.read()

    # Replace ASR config and path
    content = re.sub(r'ASR_config:\s*".*?"', 'ASR_config: "Utils/ASR/config.yml"', content)
    content = re.sub(r'ASR_path:\s*".*?"', 'ASR_path: "Utils/ASR/epoch_00200.pth"', content)

    # Replace data params
    content = re.sub(r'train_data:\s*".*?"', 'train_data: "../create-dataset/id/final_dataset/phonemized_lists/train_list_phon.txt"', content)
    content = re.sub(r'val_data:\s*".*?"', 'val_data: "../create-dataset/id/final_dataset/phonemized_lists/val_list_phon.txt"', content)
    content = re.sub(r'root_path:\s*".*?"', 'root_path: "../create-dataset/id/final_dataset/wavs"', content)
    content = re.sub(r'OOD_data:\s*".*?"', 'OOD_data: "../create-dataset/id/final_dataset/phonemized_lists/val_list_phon.txt"', content)

    with open(file_path, "w") as f:
        f.write(content)

print(f"Updated {len(yml_files)} config files.")

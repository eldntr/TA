import os
import glob
import re

configs_dir = "/home/user/TA-Eldin/TA/StyleTTS2/Configs"
yml_files = glob.glob(f"{configs_dir}/**/*.yml", recursive=True)

for file_path in yml_files:
    with open(file_path, "r") as f:
        content = f.read()

    # 1. Update load_only_params to true for all to prevent shape mismatch
    content = re.sub(r'load_only_params:\s*(true|false)', 'load_only_params: true', content, flags=re.IGNORECASE)

    if 'config_first' in file_path:
        # Update first stage configs
        content = re.sub(r'epochs_1st:\s*\d+', 'epochs_1st: 30', content)
        # Setting pretrained_model to the English checkpoint
        content = re.sub(r'pretrained_model:\s*".*?"', 'pretrained_model: "epoch_2nd_00100.pth"', content)
    else:
        # Update second stage or ft configs
        content = re.sub(r'epochs_2nd:\s*\d+', 'epochs_2nd: 30', content)
        content = re.sub(r'epochs:\s*\d+', 'epochs: 30', content)
        # Setting pretrained_model to the English checkpoint (for predictor/style encoder)
        content = re.sub(r'pretrained_model:\s*".*?"', 'pretrained_model: "epoch_2nd_00100.pth"', content)
        
        # Determine scenario name from folder (e.g., styletts2, lpep)
        scenario = os.path.basename(os.path.dirname(file_path))
        # Wait, the first stage path might be saved as first_stage.pth or epoch_1st_00030.pth
        # But we don't know exactly what name train_first.py will save. We will leave first_stage_path 
        # as what the user had, but ensure they know to check it later.
        
    with open(file_path, "w") as f:
        f.write(content)

print(f"Berhasil mengaplikasikan penyesuaian Cross-Lingual pada {len(yml_files)} file konfigurasi.")

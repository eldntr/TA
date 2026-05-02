import csv
import os
import yt_dlp

def download_audio(csv_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(csv_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            youtube_url = row.get('Link Youtube')
            channel_name = row.get('Channel')

            if not youtube_url or not channel_name:
                continue

            channel_dir_name = "".join([c for c in channel_name if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).rstrip()
            channel_path = os.path.join(output_dir, channel_dir_name)
            
            if not os.path.exists(channel_path):
                os.makedirs(channel_path)
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(channel_path, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'postprocessor_args': [
                    '-ar', '16000',
                    '-ac', '1'
                ],
                'ignoreerrors': True,
                'no_warnings': True,
                'quiet': False
            }
            
            print(f"Downloading from {channel_name}: {youtube_url}")
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
            except Exception as e:
                print(f"Failed to download {youtube_url}: {e}")

if __name__ == "__main__":
    csv_filename = "Dataset TA - Single .csv"
    output_directory = "dataset_audio"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_filename)
    out_dir_path = os.path.join(script_dir, output_directory)
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find '{csv_filename}' in {script_dir}")
    else:
        print("Starting download process...")
        download_audio(csv_path, out_dir_path)
        print("Download process completed.")

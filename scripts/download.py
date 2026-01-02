import os, subprocess, pandas as pd
from glob import glob

def download_and_process(csv_file, output_root):
    if not os.path.exists(csv_file): return print(f"[ERROR] Missing: {csv_file}")
    df = pd.read_csv(csv_file)
    os.makedirs(output_root, exist_ok=True)

    for _, row in df.iterrows():
        vid_id, link, start, end, fps = str(row['id']), row['youtube_link'], row['start_time'], row['end_time'], row['fps']
        save_dir = os.path.join(output_root, vid_id, 'images')
        if os.path.exists(save_dir) and glob(os.path.join(save_dir, "*.png")): continue
        os.makedirs(save_dir, exist_ok=True)

        print(f"Processing ID: {vid_id} | FPS: {fps}")
        dl_start, temp_vid = max(0, start - 5.0), f"temp_{vid_id}.mp4"
        
        try:
            # Download buffered segment
            subprocess.run([
                "yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
                "--download-sections", f"*{dl_start}-{end + 5.0}", "--force-keyframes-at-cuts",
                "-o", temp_vid, link
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

            # Precise trim & extract
            subprocess.run([
                "ffmpeg", "-ss", str(start - dl_start), "-i", temp_vid, "-t", str(end - start),
                "-vf", f"fps={fps},scale=iw/{row['resize_factor']}:-1", "-start_number", "0",
                "-vframes", str(int(round((end - start) * fps))), os.path.join(save_dir, "%04d.png"),
                "-y", "-loglevel", "error"
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"[SUCCESS] {vid_id}: {len(glob(os.path.join(save_dir, '*.png')))} frames.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed {vid_id}: {e}")
        finally:
            if os.path.exists(temp_vid): os.remove(temp_vid)

if __name__ == "__main__":
    download_and_process(
        csv_file='./data/dataset.csv',
        output_root='./data/splannequin'
        )

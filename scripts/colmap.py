import os
import subprocess
import argparse
import struct
from glob import glob

# --- CONFIGURATION ---
DEFAULT_ROOT = './data/splannequin'
COLMAP_BIN = "colmap"
# ---------------------

BOLD_GREEN = "\033[1;32m"
RESET = "\033[0m"

def run_command(cmd, log_file=None):
    try:
        stdout_dest = open(log_file, 'w') if log_file else subprocess.DEVNULL
        subprocess.run(cmd, check=True, stdout=stdout_dest, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed: {' '.join(cmd)}")
        if log_file: print(f"Check log: {log_file}")
        raise e
    finally:
        if log_file and stdout_dest: stdout_dest.close()

def get_best_model(sparse_dir):
    if not os.path.exists(sparse_dir): return None, 0
    best_count, best_path = -1, None
    for sub in os.listdir(sparse_dir):
        model_path = os.path.join(sparse_dir, sub)
        img_bin = os.path.join(model_path, "images.bin")
        if os.path.isdir(model_path) and os.path.exists(img_bin):
            with open(img_bin, 'rb') as f:
                count = struct.unpack('Q', f.read(8))[0]
            if count > best_count:
                best_count, best_path = count, model_path
    return best_path, best_count

def process_video_folder(base_dir, use_gpu):
    images_dir = os.path.join(base_dir, "images")
    
    # Enforce standard structure
    if not os.path.exists(images_dir):
        print(f"[SKIP] {os.path.basename(base_dir)}: No 'images' folder found.")
        return False

    db_path = os.path.join(base_dir, "database.db")
    sparse_dir = os.path.join(base_dir, "sparse")
    log_file = os.path.join(base_dir, "colmap_log.txt")
    
    # Skip if done
    best_path, _ = get_best_model(sparse_dir)
    if best_path:
        print(f"[INFO] Skipping {os.path.basename(base_dir)}, reconstruction exists.")
        return True

    print(f"\nProcessing {BOLD_GREEN}{os.path.basename(base_dir)}{RESET}...")
    os.makedirs(sparse_dir, exist_ok=True)

    # 1. Feature Extraction
    print(f"  - Extracting features...", end="", flush=True)
    run_command([
        COLMAP_BIN, "feature_extractor", "--database_path", db_path, "--image_path", images_dir,
        "--SiftExtraction.use_gpu", "1" if use_gpu else "0",
        "--ImageReader.camera_model", "SIMPLE_PINHOLE", "--ImageReader.single_camera", "1"
    ], log_file)
    print(" Done.")

    # 2. Matching
    print(f"  - Matching features...", end="", flush=True)
    run_command([
        COLMAP_BIN, "sequential_matcher", "--database_path", db_path,
        "--SiftMatching.use_gpu", "1" if use_gpu else "0",
        "--SequentialMatching.loop_detection", "1", "--SequentialMatching.overlap", "20"
    ], log_file)
    print(" Done.")

    # 3. Reconstruction
    print(f"  - Reconstructing...", end="", flush=True)
    run_command([
        COLMAP_BIN, "mapper", "--database_path", db_path,
        "--image_path", images_dir, "--output_path", sparse_dir
    ], log_file)
    print(" Done.")

    # 4. Converter
    best_path, count = get_best_model(sparse_dir)
    if best_path:
        txt_dir = best_path + "_txt"
        os.makedirs(txt_dir, exist_ok=True)
        run_command([
            COLMAP_BIN, "model_converter", "--input_path", best_path,
            "--output_path", txt_dir, "--output_type", "TXT"
        ])
        print(f"  {BOLD_GREEN}[SUCCESS]{RESET} Registered {count} images.")
    else:
        print(f"  [WARN] Reconstruction failed.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run COLMAP (requires 'images' subdir).")
    parser.add_argument("path", nargs="?", default=DEFAULT_ROOT, help="Path to video folder (must contain 'images/') OR root containing multiple videos")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"[ERROR] Path not found: {args.path}")
        return

    # LOGIC:
    # 1. Check if 'path' ITSELF is a video folder (has 'images' subdir)
    if os.path.exists(os.path.join(args.path, "images")):
        # Single mode
        process_video_folder(args.path, args.gpu)
    else:
        # Root mode: iterate subfolders
        subdirs = sorted([os.path.join(args.path, d) for d in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, d))])
        valid = [d for d in subdirs if os.path.exists(os.path.join(d, "images"))]
        
        if not valid:
            print(f"[ERROR] No valid datasets found in {args.path}")
            print(f"Structure must be: {args.path}/<video_id>/images/")
            return

        print(f"Found {len(valid)} datasets in {args.path}")
        if args.gpu: print("GPU Acceleration: ENABLED")
        
        for d in valid:
            try:
                process_video_folder(d, args.gpu)
            except Exception as e:
                print(f"[FAIL] Error processing {os.path.basename(d)}: {e}")

if __name__ == "__main__":
    main()

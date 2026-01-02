# assessment/loader.py
import cv2, os, re, concurrent.futures
import numpy as np
from typing import List, Tuple, Optional

def _log(v, m): 
    if v: 
        print(m)

def load_single(path: str) -> Optional[object]:
    img = cv2.imread(path)
    if img is None:
        print(f"[Loader] failed: {path}")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def extract_frame_index(path) -> int:
        # Coerce to string for regex safety
        try:
            path_str = os.fspath(path)  # handles str and PathLike
        except TypeError:
            path_str = str(path)
        base = os.path.basename(path_str)
        name, _ = os.path.splitext(base)
        m = re.search(r'\d+', name)
        return int(m.group()) if m else 0

def load_batch(paths: List[str], num_workers=4) -> Tuple[list, list, list]:
    images, indices, failed = [], [], []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        fut_to_path = {ex.submit(load_single, p): p for p in paths}
        for fut in concurrent.futures.as_completed(fut_to_path):
            p = fut_to_path[fut]
            img = fut.result()
            if img is None:
                failed.append(p)
            else:
                images.append(img)
                indices.append(extract_frame_index(p))
    return images, indices, failed

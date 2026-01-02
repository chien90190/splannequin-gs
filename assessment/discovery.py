# assessment/discovery.py
import os, glob
import re, math, fnmatch 
from pathlib import Path
from typing import List, Dict, Tuple, Union

_NUM = re.compile(r'(\d+(?:\.\d+)?)')  # compile once

def sort_key_for_scene(item, patterns=("bad-*", "failed-*"), regexes=(), demote=None):
    p = item if isinstance(item, (tuple, list)) else item  # accept (path, …) or path
    name = Path(p).parts[3].split('_')[0]
    demoted = (
        any(fnmatch.fnmatch(name, pat) for pat in patterns) or
        any(re.search(rx, name) for rx in regexes) or
        (demote(name) if callable(demote) else False)
    )
    m = _NUM.search(name)
    val = float(m.group(1)) if m else math.inf               # pushes non-numeric last
    return (1 if demoted else 0, val) 

def _log(v, m): 
    if v: 
        print(m)

def has_image_files(folder: str, exts=('.jpg','.jpeg','.png','.bmp','.tiff')) -> bool:
    return any(os.path.splitext(f)[1].lower() in exts for f in os.listdir(folder))

def find_sequences(dir_path: str, verbose: bool = False) -> List[str]:    
    seqs = []
    if has_image_files(dir_path):
        seqs.append(dir_path)

    for d in sorted(os.listdir(dir_path)):
        p = os.path.join(dir_path, d)
        if os.path.isdir(p) and d.startswith("t") and has_image_files(p):
            seqs.append(p)

    if len(seqs):
        _log(verbose, f"[FindSeqs] {dir_path} -> {len(seqs)} sequences")
   
    def t_key(p):
        name = os.path.basename(p)
        m = re.match(r'^t(\d+)$', name)
        return (0, int(m.group(1))) if m else (1, name)
    
    return sorted(seqs, key=t_key)

def find_image_directories(
        base_dir: str,
        mode: str,
        type_name: str = "",
        verbose: bool = False, return_sequences: bool = False,) -> Union[List[str], List[Tuple[str, List[str]]]]:
    
    _log(verbose, f"[FindDirs] base={base_dir} mode={mode} type={type_name or 'None'}")
    patterns = [
        f"{base_dir}/*/{type_name}/eval/*_*/renders",
        f"{base_dir}/*/{type_name}/eval/*_*/{mode}",
        f"{base_dir}/*/{type_name}*/eval/*_*/{mode}",
        f"{base_dir}/{type_name}*/eval/*_*/{mode}",
        f"{base_dir}/{type_name}/eval/*_*/{mode}",
        f"{base_dir}/eval/*/renders",
        f"{base_dir}/*/eval/*/renders",
        f"{base_dir}/*/*/eval/*/renders",
        f"{base_dir}/eval/*/{mode}",
        f"{base_dir}/eval/*/{mode}/renders",
        f"{base_dir}/eval/*/{mode}/renders/*",
        f"{base_dir}/eval/*/{mode}/renders/iqa",
        f"{base_dir}/*/eval/*/{mode}",
        f"{base_dir}/*/eval/*/{mode}/renders",
        f"{base_dir}/*/eval/*/{mode}/renders/iqa",
        f"{base_dir}/*/eval/*",
        f"{base_dir}/eval/*",
        f"{base_dir}/*/*/eval/*",
    ]

    found_dirs = set()
    pairs: List[Tuple[str, List[str]]] = []

    # try each pattern
    for pat in patterns:
        _log(verbose, f"[FindDirs] Trying pattern: {pat}")
        for m in glob.glob(pat, recursive=False):
            if os.path.isdir(m) and m not in found_dirs:
                seqs = find_sequences(m, verbose=verbose)
                if seqs:
                    found_dirs.add(m)
                    if return_sequences:
                        pairs.append((m, seqs))
        if found_dirs:
            break

    # if find anything
    if return_sequences:
        pairs.sort(key=lambda t: sort_key_for_scene(t[0]))
        _log(verbose, f"[FindDirs] found={len(pairs)} (with sequences)")
        return pairs
    
    out = sorted(list(found_dirs))
    _log(verbose, f"[FindDirs] found={len(out)}")
    return out

def list_images(folder: str, exts=('.jpg','.jpeg','.png','.bmp','.tiff'), verbose: bool = False) -> List[str]:
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in exts]
    files.sort()
    return files
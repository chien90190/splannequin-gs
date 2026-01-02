# assessment/writers.py
import os, csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import json

def _log(v, m): 
    if v: 
        print(m)

def write_detailed_csv(sequence_result: Dict, out_path: str, verbose: bool = False):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        
        header = ["Frame", "Frame_Index", "Score"]
        if sequence_result.get("attribute_scores"):
            header += list(sequence_result["attribute_scores"].keys())
        w.writerow(header)
        
        frame_scores = sequence_result.get("frame_scores", [])
        frame_indices = sequence_result.get("frame_indices", list(range(len(frame_scores))))
        
        for i, (score, idx) in enumerate(zip(frame_scores, frame_indices)):
            row = [i, idx, float(score)]
            if sequence_result.get("attribute_scores"):
                for attr, vals in sequence_result["attribute_scores"].items():
                    row.append(float(vals[i]) if i < len(vals) else "")
            w.writerow(row)
    # _log(verbose, f"[WriteCSV] detail summary -> {out_path}, rows={len(frame_scores)}")

def write_quality_plot(sequence_result: Dict, out_dir: str, metric_type: str, verbose: bool = False):
    if not sequence_result.get("frame_scores"):
        _log(verbose, "[Plot] skip: no scores")
        return
    os.makedirs(out_dir, exist_ok=True)
    frames = sequence_result.get("frame_indices") or list(range(len(sequence_result["frame_scores"])))
    scores = np.array(sequence_result["frame_scores"], dtype=float)
    plt.figure(figsize=(10,5))
    plt.plot(frames, scores, linewidth=1.5)
    plt.title(f"{sequence_result['sequence_name']} ({metric_type})")
    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{sequence_result['sequence_name']}_{metric_type}_quality.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    _log(verbose, f"[Plot] saved -> {out_path}")

def calculate_overall_summary(directory_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates overall summary statistics across all sequences.
    """
    sequences = directory_result.get("sequences", {})
    if not sequences:
        return {
            "overall_avg_score": 0.0,
            "overall_avg_top_50": 0.0,
            "overall_avg_bottom_50": 0.0,
            "total_sequences": 0,
            "total_frames": 0,
        }

    total_seqs = len(sequences)
    return {"overall_avg_score": sum(seq.get("avg_score", 0.0) for seq in sequences.values()) / total_seqs,
            "overall_avg_top_50": sum(seq.get("avg_top_50", 0.0) for seq in sequences.values()) / total_seqs,
            "overall_avg_bottom_50": sum(seq.get("avg_bottom_50", 0.0) for seq in sequences.values()) / total_seqs,
            "total_sequences": total_seqs,
            "total_frames": sum(seq.get("count", 0) for seq in sequences.values()),
            }

def write_summary_json(directory_result: Dict[str, Any], out_path: str, verbose: bool = False):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary_stats = calculate_overall_summary(directory_result)
    directory_result["summary"] = summary_stats
    with open(out_path, 'w') as f:
        json.dump(directory_result, f, indent=2)
    _log(verbose, f"[WriteJSON] Summary saved to -> {out_path}")


def find_numpy_types(obj, path=""):
    """
    Recursively finds and prints the paths to NumPy data types
    within a nested dictionary or list.
    """
    # If the object is a dictionary, iterate through its items
    if isinstance(obj, dict):
        for key, value in obj.items():
            # Create a path to show the location (e.g., 'data.metadata')
            new_path = f"{path}.{key}" if path else key
            find_numpy_types(value, new_path)
    # If the object is a list, iterate through its elements
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            # Create a path with the index (e.g., 'items[2]')
            new_path = f"{path}[{i}]"
            find_numpy_types(item, new_path)
    # Base case: Check if the object itself is a NumPy type
    elif isinstance(obj, (np.ndarray, np.generic)):
        print(f"Found NumPy type at path '{path}': {type(obj).__name__}")


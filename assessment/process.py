# assessment/processor.py
import numpy as np
import torch
from typing import Dict, Any, List

def _log(v, m): 
    if v: 
        print(m)

def compute_sequence_metrics(frame_scores: List[float], frame_indices: List[int], verbose: bool=False) -> Dict[str, Any]:
    """
    Computes sequence metrics, including top and bottom 50% actual frame indices.
    """
    if not frame_scores:
        return {
            "avg_score": 0.0, "min_score": 0.0, "max_score": 0.0,
            "count": 0, "avg_top_50": 0.0, "avg_bottom_50": 0.0,
            "top_50_indices": [], "bottom_50_indices": []
        }
    scores_arr = np.array(frame_scores, dtype=float)
    indices_arr = np.array(frame_indices)

    sorted_order_indices = np.argsort(scores_arr)
    sorted_scores = scores_arr[sorted_order_indices]
    sorted_frame_indices = indices_arr[sorted_order_indices]
    
    # Find the midpoint to split the data into top and bottom 50%
    midpoint = len(sorted_scores) // 2
    
    # Separate the scores and the actual indices into two halves
    bottom_half_scores = sorted_scores[:midpoint]
    top_half_scores = sorted_scores[midpoint:]
    
    bottom_50_indices = sorted_frame_indices[:midpoint]
    top_50_indices = sorted_frame_indices[midpoint:]

    worst_score_frame = indices_arr[np.argmin(scores_arr)]

    return {"avg_score": float(np.mean(scores_arr)),
            "min_score": float(np.min(scores_arr)),
            "max_score": float(np.max(scores_arr)),
            "count": len(frame_scores),
            "avg_top_50": float(top_half_scores.mean()) if top_half_scores.size > 0 else 0.0,
            "avg_bottom_50": float(bottom_half_scores.mean()) if bottom_half_scores.size > 0 else 0.0,
            "top_50_indices": top_50_indices.tolist(),
            "bottom_50_indices": bottom_50_indices.tolist(),
            'worst_score_frame': int(worst_score_frame),
            }

def weighted_average(sequences: List[Dict[str, float]], verbose: bool = False):
    total, n = 0.0, 0
    for s in sequences:
        total += s["avg_score"] * s["count"]
        n += s["count"]
    avg = (total / n) if n else 0.0
    _log(verbose, f"[WeightedAvg] total_frames={n} overall={avg:.4f}")
    return avg, n


def _to_floats(x):
    """Convert anything to list of floats"""
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().view(-1).numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).reshape(-1).tolist()
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return [float(x)]

def format_output(raw):
    """Clean up model output to consistent format"""
    if not isinstance(raw, dict):
        raw = {"score": raw}
    
    result = {"score": _to_floats(raw.get("score", []))}
    
    if "attribute_scores" in raw:
        result["attribute_scores"] = {k: _to_floats(v) for k, v in raw["attribute_scores"].items()}
    
    if "saliency_map" in raw:
        result["saliency_map"] = raw["saliency_map"]
    
    return result
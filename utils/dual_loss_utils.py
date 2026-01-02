import torch
import torch.nn as nn
import json
import os
import datetime

def format_sci(val: float) -> str:
    """Compact scientific notation; e.g., 1.23e5, 1e6."""
    s = f"{val:.3e}".replace('e+0', 'e').replace('e+', 'e')
    mant_str, exp_str = s.split('e')
    mant, exp = float(mant_str), int(exp_str)
    if abs(abs(mant) - 1.0) < 1e-10:
        return f"{'-' if mant < 0 else ''}1e{exp}"
    return f"{mant_str.rstrip('0').rstrip('.')}e{exp}"

def save_training_summary(timer, opt, args):
    total_time = timer.get_elapsed_time()
    total_iterations = opt.coarse_iterations + opt.iterations
    
    training_summary = {
        "training_completion_time": datetime.datetime.now().isoformat(),
        "total_training_time_seconds": round(total_time, 2),
        "total_training_time_hours": round(total_time / 3600, 2),
        "total_iterations": total_iterations,
        "average_time_per_iteration_seconds": round(total_time / total_iterations, 3),
        "coarse_iterations": opt.coarse_iterations,
        "fine_iterations": opt.iterations,
        "gpu_name": torch.cuda.get_device_name(),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
        "model_path": args.model_path,
        "batch_size": opt.batch_size if hasattr(opt, 'batch_size') else None
    }
    # Save to JSON file
    json_path = os.path.join(args.model_path, "training_summary.json")
    with open(json_path, 'w') as f:
        json.dump(training_summary, f, indent=4)
    
    print(f"Training summary saved to: {json_path}")
    return training_summary


class DualLoss2View(nn.Module):
    """
        2-view temporal consistency loss for 3D Gaussian Splatting.
        
        Input tensors (data, visibility_mask, times) must be sorted by timestamp 
        in ascending order along the first dimension (index 0 = earlier time, index 1 = later time).
        
        Enforces temporal consistency between Gaussian parameters across two views by:
        - Anchoring hidden Gaussians to visible ones (supervised → unsupervised)
        - Anchoring occluded Gaussians to visible ones  
        - Applying equality constraints for same-visibility pairs within temporal threshold
        
        Key design principles:
        - Uses .detach() to prevent unsupervised states from influencing supervised ones
        - Switches from L2 to L1 loss at specified iterations for better convergence
        - Applies confidence weighting based on temporal distance
        - Assumes temporal ordering: times[0] ≤ times[1] for proper calculation
    """
    def __init__(self, opt):
        super(DualLoss2View, self).__init__()
        
        # Iteration thresholds for L2→L1 switching (L1 is more robust in later training)
        self.l1_hidden_from_iter = opt.l1_hidden_from_iter
        self.l1_occlusion_from_iter = opt.l1_occlusion_from_iter
        
        # Confidence parameters for temporal weighting
        if hasattr(opt, 'num_frames'):
            self.num_frames = opt.num_frames
            self.equality_threshold = 10.0 / self.num_frames
        else:
            self.num_frames = None
            print("Warning: 'opt' object has no attribute 'num_frames'. Setting self.num_frames to None.")
        self.hidden_steepness = opt.hidden_steepness      
        self.occlude_steepness = opt.occlusion_steepness  
        
        # Temporal threshold for equality constraints (normalized by sequence length)
        self.use_equality = opt.use_equality
        if not self.use_equality: 
            print('Not using equality mode.')

    def update(self, num_frames):
        self.num_frames = num_frames
        self.equality_threshold = 10.0 / self.num_frames
    
    def compute_hidden_loss(self, iteration, times, deforms, visibility_counts, threshold):
        
        # Identify hidden Gaussians (not visible in second view)
        # visibility patterns: [True, False] or [False, False]
        hidden_gaussians = (visibility_counts[1] == False)  
        
        # Compute percentage for monitoring/thresholding
        h_percent = hidden_gaussians.float().mean()
        if h_percent > threshold:
            with torch.no_grad():
                style = "l1" if iteration > self.l1_hidden_from_iter else "l2"
                confidences = self._compute_confidence(times, self.hidden_steepness)     
            return self._hidden_loss(deforms, times, visibility_counts, hidden_gaussians, confidences, style), h_percent
        else:
            return None, h_percent
    
    def compute_occlusion_loss(self, iteration, times, deforms, visibility_counts, grad_counts, threshold):
        
        # Identify occluded Gaussians: in frustum but no gradients in first view
        # gradient patterns: [False, True] or [False, False]
        occluded_gaussians = (visibility_counts.sum(dim=0) > 1) & (grad_counts[0] == False)
        
        o_percent = occluded_gaussians.float().mean()
        if o_percent > threshold:
            with torch.no_grad():
                style = "l1" if iteration > self.l1_occlusion_from_iter else "l2"
                confidences = self._compute_confidence(times, self.occlude_steepness)
            return self._occluded_loss(deforms, times, grad_counts, occluded_gaussians, confidences, style), o_percent
        else:
            return None, o_percent
    
    def _hidden_loss(self, data, times, visibility_counts, valid_gaussians, confidence, loss_type):
        """
        Core hidden loss computation with two cases:
        
        Case 1: [Visible, Hidden] - Anchor hidden states to visible states
        Case 2: [Hidden, Hidden] - Apply equality constraint if temporally close
        
        Args:
            data (Tensor): Deformation parameters [2, num_gaussians, features]
            times (Tensor): Timestamps [2]
            visibility_counts (Tensor): Boolean visibility [2, num_gaussians]  
            valid_gaussians (Tensor): Boolean mask of Gaussians to process
        """
        # Filter to valid Gaussians only
        data = data[:, valid_gaussians, :]
        visibility_counts = visibility_counts[:, valid_gaussians].detach()
        
        assert data.shape[0] == 2, "This function expects exactly 2 views"
        
        vis_first = visibility_counts[0]
        vis_second = visibility_counts[1]
        
        total_loss = torch.tensor(0.0, device=data.device, dtype=data.dtype)
        
        # Case 1: [True, False] - Use visible state to supervise hidden state
        target_mask = vis_first & (~vis_second)
        if target_mask.any():
            # .detach() prevents hidden states from influencing visible ones
            # (visible states are supervised by ground truth reconstruction loss)
            ref_data = data[0][target_mask].detach()  # Visible state (stable anchor)
            target_data = data[1][target_mask]        # Hidden state (to be optimized)
            view_confidence = confidence[0]           # Weight based on first view confidence
            
            loss_tensor = self._apply_loss_function(target_data, ref_data, loss_type)
            total_loss += (loss_tensor * view_confidence).mean()
        
        # Case 2: [False, False] - Both hidden, enforce temporal smoothness
        # If frames are temporally close, hidden states should be similar
        if self.use_equality:
            same_vis_mask = (~vis_first) & (~vis_second)
            if same_vis_mask.any() and abs(times[1] - times[0]) < self.equality_threshold:
                ref_data = data[0][same_vis_mask]           # Use first frame as reference
                target_data = data[1][same_vis_mask]        # Second frame follows first
            
                equality_loss = self._apply_loss_function(target_data, ref_data, loss_type)
                total_loss += equality_loss.mean()
        
        return total_loss
    
    def _occluded_loss(self, data, times, grad_counts, valid_gaussians, confidence, loss_type):
        """
        Core occlusion loss computation with two cases:
        
        Case 1: [Occluded, Visible] - Anchor occluded states to visible states  
        Case 2: [Occluded, Occluded] - Apply equality constraint if temporally close
        
        Args:
            data (Tensor): Sorted Deformation parameters [2, num_gaussians, features]
            times (Tensor): Sorted Timestamps [2]
            grad_counts (Tensor): Sorted Boolean gradients [2, num_gaussians]
            valid_gaussians (Tensor): Sorted Boolean mask of Gaussians to process [num_gaussians]
        """
        data = data[:, valid_gaussians, :]
        grad_counts = grad_counts[:, valid_gaussians].detach()
        
        assert data.shape[0] == 2, "This function expects exactly 2 views"
        
        grad_first = grad_counts[0]
        grad_second = grad_counts[1]
        
        total_loss = torch.tensor(0.0, device=data.device, dtype=data.dtype)
        
        # Case 1: [False, True] - Use visible state to supervise occluded state
        # Weight based on first view confidence
        target_mask = (~grad_first) & grad_second
        if target_mask.any():
            ref_data = data[1][target_mask].detach()  # Visible state (stable anchor)
            target_data = data[0][target_mask]
            view_confidence = confidence[0]          
            
            loss_tensor = self._apply_loss_function(target_data, ref_data, loss_type)
            total_loss += (loss_tensor * view_confidence).mean()
        
        # Case 2: [False, False] - Both occluded, enforce temporal smoothness. First frame follows second.
        if self.use_equality:
            same_vis_mask = (~grad_first) & (~grad_second)
            if same_vis_mask.any() and abs(times[1] - times[0]) < self.equality_threshold:
                ref_data = data[1][same_vis_mask].detach()
                target_data = data[0][same_vis_mask]        
            
                equality_loss = self._apply_loss_function(target_data, ref_data, loss_type)
                total_loss += equality_loss.mean()
        
        return total_loss
    
    def _compute_confidence(self, times, steepness):
        """Frames closer to the maximum timestamp receive higher confidence,"""
        max_val = times.detach().max()
        steps_away = times - max_val
        return torch.exp(steepness * steps_away)
    
    def _apply_loss_function(self, target, reference, loss_type):
        """Apply L1 or L2 loss element-wise between target and reference tensors."""
        if loss_type == "l2":
            return (target - reference) ** 2
        else:  # l1
            return torch.abs(target - reference)

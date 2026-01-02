import torch
import os
import json
import concurrent.futures
import numpy as np
import imageio
import torchvision
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import svd
import numpy as np
import copy
import time
import csv

from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams

# ================================
# Utility Functions
# ================================
def multithread_write(image_list, path, llffhold=8):
    """Write images to disk using multithreading with simple sequential naming."""
    os.makedirs(path, exist_ok=True)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    
    def write_image(image, count, path, llffhold):
        try:
            if count % llffhold != 0:
                return count, False
            
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image)
            if len(image.shape) == 3 and image.shape[2] in [1, 3, 4]:
                image = image.permute(2, 0, 1)
            if image.max() > 1.0:
                image = image / 255.0
                
            torchvision.utils.save_image(
                image, os.path.join(path, '{0:04d}'.format(count) + ".png")
            )
            return count, True
        except Exception as e:
            print(f"Error saving image {count}: {e}")
            return count, False
    
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path, llffhold))
    
    results = [task.result() for task in tasks]
    executor.shutdown()
    
    saved_count = sum(1 for _, status in results if status)
    print(f'>> Saved {saved_count:,} (every {llffhold}th from {len(image_list)} total) images to {path}.')

def extract_from_json_file(file_path):
    """Extract configuration from JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    key_setting = "parameters" if "parameters" in data else "settings"
    
    # Try all possible locations for fps
    fps_locations = [
        (data.get(key_setting, {}), "target_fps"),
        (data, "segment_fps")
    ]
    
    target_fps = None
    for container, key in fps_locations:
        if key in container:
            target_fps = container[key]
            break
    
    if target_fps is None:
        raise ValueError("Could not find fps in JSON")
    
    # Try all possible locations for frame count
    frame_locations = [
        (data.get(key_setting, {}), "total_frames"),
        (data.get('output', {}), "total_frames"),
        (data, "extracted_frames")
    ]
    
    total_frames = None
    for container, key in frame_locations:
        if key in container:
            total_frames = container[key]
            break
    
    if total_frames is None:
        raise ValueError("Could not find frame count in JSON")
    
    return target_fps, total_frames


def to8b(x):
    """Convert tensor to 8-bit numpy array."""
    return (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

def get_time_indices(description, cameras, time_index_str=None):
    """Helper function to determine which time indices to render."""
    
    # Default to first frame
    time_indices = np.array([0])
    
    if any(keyword in description for keyword in ["time-sweeping", "fixed-time"]):
        # Create indices every 8 frames, skipping the first
        additional_indices = np.arange(0, len(cameras), 8)[1:]
        time_indices = np.concatenate((time_indices, additional_indices))

        if time_index_str:
            selected_indices = np.fromstring(time_index_str, dtype=int, sep=',')
            return time_indices[selected_indices]
    
    return time_indices

# ================================
# Camera Functions
# ================================
def view_interpolation(idx, views, frames):
    """Simple, clean interpolation handling both M==N and M>N cases."""
    total_views = len(views)
    if total_views < 2:
        raise ValueError("Need at least two views for interpolation.")

    idx = max(0, min(idx, frames - 1))

    # When frames == views, direct 1:1 mapping (no interpolation needed)
    if frames == total_views:
        view = views[idx]
        return view.R, view.T, idx, idx

    # Standard interpolation for M > N
    ratio = idx / (frames - 1) if frames > 1 else 0.0
    query_idx = ratio * (total_views - 1)

    begin_idx = int(np.floor(query_idx))
    end_idx = min(begin_idx + 1, total_views - 1)
    interp_ratio = query_idx - begin_idx

    view_begin = views[begin_idx]
    view_end = views[end_idx]

    R_cur = (1 - interp_ratio) * view_begin.R + interp_ratio * view_end.R
    T_cur = (1 - interp_ratio) * view_begin.T + interp_ratio * view_end.T

    return R_cur, T_cur, idx, begin_idx

def smooth_trajectory(views, sigma=2.0):
    """Apply Gaussian smoothing to camera trajectory."""
    if len(views) < 3:
        return views
    
    # Extract R and T
    Rs = np.array([view.R for view in views])  # (N, 3, 3)
    Ts = np.array([view.T for view in views])  # (N, 3)
    
    # Smooth each component
    Rs_flat = Rs.reshape(len(views), -1)  # (N, 9)
    Rs_smooth = gaussian_filter1d(Rs_flat, sigma=sigma, axis=0)
    Ts_smooth = gaussian_filter1d(Ts, sigma=sigma, axis=0)
    
    # Reshape back and orthonormalize rotations
    Rs_smooth = Rs_smooth.reshape(len(views), 3, 3)
    for i in range(len(views)):
        U, _, Vt = svd(Rs_smooth[i])
        Rs_smooth[i] = U @ Vt

    return Rs_smooth, Ts_smooth

# ================================
# Core Rendering Functions
# ================================
def render_set(views, gaussians, pipeline, background, cam_type, frames=300, time_index=None):
    """Render standard interpolated sequence."""
    rendered_list = []
    gt_list = []

    time1 = time.time()
    for idx, view in enumerate(tqdm(views, desc="Rendering test views")):
        render_pkg = render(view, gaussians, pipeline, background, cam_type=cam_type)
        gt = view.original_image[0:3, :, :] if cam_type != "PanopticSports" else view['image'].cuda()
        rendering = render_pkg["render"]
        gt_list.append(gt)
        rendered_list.append(rendering)
    time2 = time.time()

    render_fps = len(rendered_list)/ (time2 - time1) if len(rendered_list) > 0 else 0
    print(f'Test FPS: \033[1;35m{render_fps:.5f}\033[0m, #G: {gaussians.get_xyz.shape[0]}, {len(rendered_list)} images.')
    
    return rendered_list, gt_list, render_fps, None

def render_fixed_time(views, gaussians, pipeline, background, cam_type, time_index=0, frames=300, snap_idx = 100):
    """Render sequence with fixed time, moving camera."""
    rendered_list = []
    gt_list = []
    t_list = []
    snapshot = torch.zeros(gaussians.get_xyz.shape[0]).cuda()
    
    xyz = gaussians.get_xyz
    time_input = torch.tensor(views[time_index].time, dtype=torch.float32).to(xyz.device).repeat(xyz.shape[0], 1)
    deformed = gaussians._deformation(xyz.detach(), gaussians._scaling, gaussians._rotation, gaussians._opacity, gaussians.get_features, time_input)

    pbar = tqdm(range(frames), desc=f"Fixed-time rendering (t={time_index})")
    for idx in pbar:
        view = views[idx]

        torch.cuda.synchronize()
        t_start = time.time()

        render_pkg = render(view, gaussians, pipeline, background, cam_type=cam_type, deformed=deformed)
        rendering = render_pkg["render"]

        torch.cuda.synchronize()
        t_end = time.time()
        t_list.append(t_end - t_start)

        gt = view.original_image[0:3, :, :] if cam_type != "PanopticSports" else view['image'].cuda()
        gt_list.append(gt)
        rendered_list.append(rendering)

    t = np.array(t_list[5:])
    render_fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{render_fps:.5f}\033[0m, #G: {gaussians.get_xyz.shape[0]}, {len(rendered_list)} images.')
    return rendered_list, gt_list, render_fps, snapshot

def render_fixed_view(views, gaussians, pipeline, background, cam_type, time_index=0, frames=300):
    """Render sequence with fixed time, moving camera."""
    rendered_list = []
    gt_list = []
    view = views[time_index]
    t_list = []

    for idx in tqdm(range(frames), desc=f"Fixed-time rendering (t={time_index})"):
        torch.cuda.synchronize()
        t_start = time.time()

        view.time = idx / (frames-1)
        render_pkg = render(view, gaussians, pipeline, background, cam_type=cam_type)
        
        torch.cuda.synchronize()
        t_end = time.time()
        t_list.append(t_end - t_start)

        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :] if cam_type != "PanopticSports" else view['image'].cuda()
        gt_list.append(gt)
        rendered_list.append(rendering)

    t = np.array(t_list[5:])
    render_fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{render_fps:.5f}\033[0m, #G: {gaussians.get_xyz.shape[0]}, {len(rendered_list)} images.')
    return rendered_list, gt_list, render_fps, None

def render_smooth(views, gaussians, pipeline, background, cam_type, frames=300, time_index=None, sigma=2.0):
    """Render sequence with smoothed camera trajectory."""
    print(f"Smoothing trajectory with sigma={sigma}")
    
    Rs_smooth, Ts_smooth = smooth_trajectory(views, sigma=sigma)
    fixed_time = views[time_index].time
    rendered_list = []
    gt_list = []
    
    time1 = time.time()
    for idx, view in enumerate(tqdm(views, desc="Smooth rendering")):
        view.time = fixed_time
        view.reset_extrinsic(Rs_smooth[idx], Ts_smooth[idx])

        render_pkg = render(view, gaussians, pipeline, background, cam_type=cam_type)
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :] if cam_type != "PanopticSports" else view['image'].cuda()
        gt_list.append(gt)
        rendered_list.append(rendering)
    time2 = time.time()

    render_fps = len(rendered_list) / (time2 - time1) if len(rendered_list) > 1 else 0
    print(f'Smooth FPS: \033[1;35m{render_fps:.5f}\033[0m, #G: {gaussians.get_xyz.shape[0]}, {len(rendered_list)} images.')
    
    return rendered_list, gt_list, render_fps, None

# ================================
# Output Functions
# ================================
def setup_output_paths(model_path, name, iteration, mode, time_index=None):
    """Set up output paths for renders and videos."""
    base_dir = os.path.join(model_path, name, f"ours_{iteration}")
    video_dir = os.path.join(base_dir, mode)
    
    render_path = os.path.join(video_dir, "renders")
    if mode != "normal" and time_index is not None:
        render_path = os.path.join(render_path, f"t{time_index}")
    return video_dir, render_path

# ================================
# Main Orchestrator
# ================================
def render_sequences(dataset, hyperparam, gaussian_stage, iteration, pipeline, modes=["normal"], time_index_str=None, fps=30, frames=300):
    """Main orchestrator function similar to render_sets in render2.py"""
    
    with torch.no_grad():
        dataset.eval = False
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_stage=gaussian_stage)
        eval_cameras = scene.getTrainCameras()
        cam_type = scene.dataset_type
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Map descriptions to render functions
        render_functions = {
            "normal": render_set,
            "fixed-time-view": render_fixed_time,
            "time-sweeping-view": render_fixed_view,
            "fixed-time-smooth": render_smooth
        }
        
        for mode in modes:
            log_path = os.path.join(dataset.model_path, "eval", f"ours_{scene.loaded_iter}", mode, f"fps.csv")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w', newline='') as fps_log:
                csv.writer(fps_log).writerow(['VideoName', 'FPS'])

            print(f"Rendering mode: {mode}")
            render_func = render_functions[mode]
            
            # Get time indices
            time_indices = get_time_indices(mode, eval_cameras, time_index_str)
            fps_list = []
            for time_idx in time_indices:
                
                rendered_list, gt_list, render_fps, snapshot = render_func(eval_cameras, gaussians, pipeline, background, cam_type, frames=frames, time_index=time_idx)
                renderings = [to8b(img).transpose(1, 2, 0) for img in rendered_list]

                video_dir, render_path = setup_output_paths(dataset.model_path, "eval", scene.loaded_iter, mode, time_idx)
                video_name = "video.mp4" if mode == "normal" else f"t{time_idx}({time_idx / fps:.1f}s)_{mode}.mp4"
                
                multithread_write(renderings, render_path, llffhold=1)
                # if snapshot is not None: scene.save_masked(video_dir + "/masked.ply", snapshot)
                os.makedirs(video_dir, exist_ok=True)
                imageio.mimwrite(os.path.join(video_dir, video_name), np.stack(renderings, 0), fps=fps, quality=8, macro_block_size=4)
                print(f">> Saved {video_name}.")
                
                fps_list.append(render_fps)
                with open(log_path, 'a', newline='') as fps_log:
                    csv.writer(fps_log).writerow([video_name, render_fps])

if __name__ == "__main__":
    parser = ArgumentParser(description="Gaussian Splatting Rendering Script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)

    parser.add_argument("--iteration", default=-1, type=int, help="Model iteration to use for rendering (-1 for latest)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--configs", type=str, help="Path to configuration file")
    parser.add_argument("--gaussian_stage", default="", type=str, help="Specify the stage of the Gaussian model")
    parser.add_argument("--time_index", type=str, default="", help="Specify a single index or a comma-separated list of indices for time manipulation")
    parser.add_argument("--render_modes", type=str, nargs='+', default=["normal"], help="List of rendering modes to use")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second for video rendering (e.g., 30, 29.97)")
    parser.add_argument("--total_frames", type=int, default=300, help="Total number of frames to render")

    args = get_combined_args(parser)
    print(f"Rendering {args.model_path}")

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    
    # Initialize system state
    safe_state(args.quiet)
    
    # Load dataset configuration if available
    data_stats_path = os.path.join(args.source_path, "config.json")
    if os.path.exists(data_stats_path):
        fps, src_total_frames = extract_from_json_file(data_stats_path)
        args.total_frames = src_total_frames
        args.fps = fps
    
    # Render with specified parameters
    render_sequences(
        model.extract(args),
        hyperparam.extract(args), 
        args.gaussian_stage,
        args.iteration,
        pipeline.extract(args),
        modes=args.render_modes,
        time_index_str=args.time_index,
        fps=args.fps,
        frames=args.total_frames
    )

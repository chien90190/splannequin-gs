import os, sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import joblib
import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import slic
from skimage.util import img_as_float
from transformers import CLIPProcessor, CLIPModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Linear

from . import BASNet
from datasets import data_transforms
from .u2net import U2NET
from .FullVggCompositionNet import FullVggCompositionNet as CompositionNet
import torchvision.transforms as transforms
from typing import List, Dict, Union


# Constants for image processing
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
DEFAULT_INPUT_SIZE = (224, 224)


class DRFISaliencyModel:
    def __init__(self, rf_model_path="drfi_rf_model.pkl", device="cuda"):
        self.rf = joblib.load(rf_model_path)
        self.num_segments = 200
        self.compactness = 20
        self.sigma = 1.0
        self.device = device

    def detect_saliency(self, image):
        image = img_as_float(image)
        segments = slic(image, n_segments=self.num_segments,
                       compactness=self.compactness, sigma=self.sigma)
        features = self._extract_features(image, segments)
        saliency_scores = self.rf.predict(features)
        
        saliency_map = np.zeros(image.shape[:2])
        for i in np.unique(segments):
            saliency_map[segments == i] = saliency_scores[i]
            
        return (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

    def process_image(self, img):
        if isinstance(img, str):
            img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        elif isinstance(img, Image.Image):
            img = np.array(img)
        return img

    def __call__(self, img):
        """
        Modified to match the expected interface format.
        
        Args:
            img: Input image or batch of images
            
        Returns:
            dict: Contains 'score' and 'saliency_map' keys
        """
        # Handle batched input
        if isinstance(img, np.ndarray) and img.ndim == 4:
            batch_size = img.shape[0]
            scores = []
            saliency_maps = []
            
            for i in range(batch_size):
                single_img = self.process_image(img[i])
                saliency_map = self.detect_saliency(single_img)
                max_saliency = np.max(saliency_map)
                scores.append(max_saliency)
                saliency_maps.append(saliency_map)
            
            # Convert to tensors
            score_tensor = torch.tensor(scores, device=self.device)
            saliency_tensor = torch.tensor(np.stack(saliency_maps), device=self.device)
            # Add channel dimension for consistency
            saliency_tensor = saliency_tensor.unsqueeze(-1).repeat(1, 1, 1, 3)
            
            return {
                'score': score_tensor,
                'saliency_map': saliency_tensor
            }
        else:
            # Single image
            img = self.process_image(img)
            saliency_map = self.detect_saliency(img)
            max_saliency = np.max(saliency_map)
            
            # Convert to tensors
            score_tensor = torch.tensor([max_saliency], device=self.device)
            saliency_tensor = torch.tensor(saliency_map, device=self.device)
            # Add batch and channel dimensions
            saliency_tensor = saliency_tensor.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
            
            return {
                'score': score_tensor,
                'saliency_map': saliency_tensor
            }
        




class AestheticScorer:
    def __init__(self, device="cuda", model_path=None, l1=1024, l2=512, gpu_id=0):
        """
        Initialize the aesthetic scorer model.
        
        Args:
            device: Device to run the model on ("cuda" or "cpu")
            model_path: Path to the model checkpoint
            l1: Size of first linear layer
            l2: Size of second linear layer
            gpu_id: GPU ID to use
        """
        self.device = device
        self.model_path = model_path
        print(f"Loading aesthetic model from: {self.model_path}.")
        self.l1 = l1
        self.l2 = l2
        self.gpu_id = gpu_id
        
        # Initialize model and transforms
        self.single_pass_net = self._init_model()
        self.val_transform = self._init_transform()

    def _init_model(self):
        """Initialize the aesthetic model."""
        print("Initializing AESTHETIC Model")
        
        # Check if model file exists
        if not os.path.isfile(self.model_path):
            print(f"Aesthetic Model {self.model_path} does not exist, exiting")
            sys.exit(-1)
        
        # # Import necessary model architectures
        from .FullVggCompositionNet import FullVggCompositionNet as CompositionNet
        from .SiameseNet import SiameseNet
        
        # Initialize the model
        single_pass_net = CompositionNet(pretrained=False, LinearSize1=self.l1, LinearSize2=self.l2)
        siamese_net = SiameseNet(single_pass_net)
        
        # Load checkpoint
        # ckpt = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        ckpt = torch.load(self.model_path, map_location=lambda storage, loc: storage, weights_only=False)
        model_state_dict = ckpt['state_dict']
        siamese_net.load_state_dict(model_state_dict)
        
        # Set device
        if torch.cuda.device_count() > 0 and self.device == "cuda":
            torch.cuda.set_device(int(self.gpu_id))
            single_pass_net.to(self.device)
        
        # Set to evaluation mode
        single_pass_net.eval()
        
        return single_pass_net

    def _init_transform(self):
        """Initialize the image transformation pipeline."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _convert_to_pil(self, image):
        """Convert various image formats to PIL Image."""
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:  # (C, H, W)
                image = image.permute(1, 2, 0)  # Convert to (H, W, C)
                
            if image.dtype == torch.uint8:
                return Image.fromarray(image.cpu().numpy())
            else:
                # For float tensors, assert they're in [0, 1] range
                self._assert_tensor_range(image, 0.0, 1.0, "Input image tensor")
                # Convert to uint8 for PIL
                return Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
        elif isinstance(image, Image.Image):
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def preprocess_single_image(self, image):
        """Preprocess a single image for the model."""
        pil_image = self._convert_to_pil(image)
        tensor_image = self.val_transform(pil_image).unsqueeze(0)
        
        return tensor_image.to(self.device)

    def preprocess_batch(self, batch):
        """Preprocess a batch of images."""
        # print(f"Input batch shape: {batch.shape}.")
        # print(f" Input value range: {torch.min(batch[0])} to {torch.max(batch[0])}")
        if isinstance(batch, list):
            return self._preprocess_list_batch(batch)
        elif isinstance(batch, np.ndarray):
            return self._preprocess_numpy_batch(batch)
        elif isinstance(batch, torch.Tensor):
            return self._preprocess_tensor_batch(batch)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _preprocess_list_batch(self, batch):
        """Preprocess a list of images."""
        processed_images = []
        for image in batch:
            processed_images.append(self.preprocess_single_image(image))
        return torch.cat(processed_images, dim=0)

    def _preprocess_numpy_batch(self, batch):
        """Preprocess a numpy batch of images (B, H, W, C)."""
        processed_images = []
        for i in range(batch.shape[0]):
            processed_images.append(self.preprocess_single_image(batch[i]))
        return torch.cat(processed_images, dim=0)

    def _preprocess_tensor_batch(self, batch):
        """Preprocess a tensor batch of images."""
        processed_images = []
        for i in range(batch.shape[0]):
            processed_images.append(self.preprocess_single_image(batch[i]))
        return torch.cat(processed_images, dim=0)

    def create_dummy_saliency_map(self, batch):
        """Create a dummy saliency map for compatibility with other models."""
        # Determine batch size and dimensions
        if isinstance(batch, list):
            batch_size = len(batch)
            h, w = batch[0].shape[:2] if isinstance(batch[0], np.ndarray) else batch[0].shape[1:3]
        elif isinstance(batch, np.ndarray):
            batch_size, h, w = batch.shape[:3]
        elif isinstance(batch, torch.Tensor):
            batch_size, h, w = batch.shape[:3]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        
        # Create dummy saliency map (all ones)
        saliency_map = torch.ones(batch_size, h, w, 3, device=self.device)
        return saliency_map

    def __call__(self, batch):
        """
        Args: batch: Batch of images as list, numpy array (B, H, W, C), or torch tensor 
        Returns: dict: Contains 'score' and 'saliency_map' keys
        """
        # Score the images
        with torch.no_grad():
            processed_batch = self.preprocess_batch(batch)
            scores = self.single_pass_net(processed_batch)
        
        # Create dummy saliency map
        saliency_map = self.create_dummy_saliency_map(batch)
        return {
            'score': scores,
            'saliency_map': saliency_map
        }


class U2NetWrapper:
    def __init__(self, model_path: str = "u2net.pth", device: str = "cuda",
                 image_size=320, scoring_method: str = 'mean'):
        """
        Initialize U2NET model wrapper.

        Args:
            model_path: Path to the pre-trained model weights
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.device = torch.device(device)
        self.model = U2NET(3, 1)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.image_size=image_size
        self.scoring_method = scoring_method
        print(f'Use {self.scoring_method} for each image. [mean, threshold, top_k, weighted_mean, center_bias]')

    @staticmethod
    def normPRED(d):
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d-mi)/(ma-mi)

        return dn
    
    def _transform(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Process a batched CUDA ByteTensor for U2Net
        Each image is resized to 320×320 using GPU-based interpolation.

        Args: imgs: Input tensor (N, H, W, C) or (H, W, C) if a single image.
        """
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs).permute(2, 0, 1).unsqueeze(0)
        elif imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        
        x = imgs.float() / 255.0 # Convert from ByteTensor to float and normalize to [0-1]
        x = x.permute(0, 3, 1, 2) # Convert from [B, H, W, C] to [B, C, H, W]
        x = F.interpolate(x, 
                          size=(self.image_size, 
                                self.image_size
                                ), 
                          mode='bilinear', 
                          align_corners=False
                          )
        return (x - self.mean) / self.std
    
    def _score_interpolation(self, saliency_map, method='mean', threshold=0.5, top_k=None):
        """
        Interpolate the saliency map to produce a single score.

        Args:
            saliency_map (torch.Tensor): The saliency map produced by U2Net (B, H, W)
            method (str): The method to use for score interpolation. Options:
                - 'mean': Average saliency across the entire map
                - 'threshold': Percentage of pixels above a threshold
                - 'top_k': Average of the top K% most salient pixels
                - 'weighted_mean': Weighted average based on saliency values
                - 'center_bias': Weighted average with center bias
            threshold (float): Threshold value for 'threshold' method (0-1)
            top_k (float): Percentage of top pixels to consider for 'top_k' method (0-100)

        Returns:
            torch.Tensor: Interpolated scores (B,)
        """
        if method == 'mean':
            return saliency_map.mean(dim=[1, 2])

        elif method == 'threshold':
            return (saliency_map > threshold).float().mean(dim=[1, 2])

        elif method == 'top_k':
            if top_k is None or top_k <= 0 or top_k > 100:
                raise ValueError("top_k must be a percentage between 0 and 100")
            k = int(saliency_map.numel() * top_k / 100)
            top_values, _ = torch.topk(saliency_map.view(saliency_map.size(0), -1), k, dim=1)
            return top_values.mean(dim=1)

        elif method == 'weighted_mean':
            weights = saliency_map / saliency_map.sum(dim=[1, 2], keepdim=True)
            return (saliency_map * weights).sum(dim=[1, 2])

        elif method == 'center_bias':
            h, w = saliency_map.shape[1:]
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
            center_y, center_x = h // 2, w // 2
            distance = torch.sqrt((y - center_y)**2 + (x - center_x)**2).to(saliency_map.device)
            max_distance = torch.sqrt(torch.tensor(center_y**2 + center_x**2)).to(saliency_map.device)
            center_weights = 1 - (distance / max_distance)
            center_weights = center_weights.unsqueeze(0).expand_as(saliency_map)
            weighted_saliency = saliency_map * center_weights
            return weighted_saliency.sum(dim=[1, 2]) / center_weights.sum(dim=[1, 2])

        else:
            raise ValueError(f"Unknown method: {method}")

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        elif img.dim() == 3:
            img = img.unsqueeze(0)

        img = self._transform(img)
        d1, d2, d3, d4, d5, d6, d7 = self.model(img)
        pred = d1[:, 0, :, :]
        pred = self.normPRED(pred)

        # Use the new _score_interpolation method
        score = self._score_interpolation(pred, method='weighted_mean')

        return {
            'score': score,
            'saliency_map': pred
        }


    def __call__(self, img):
        # Process the batch
        img = self._transform(img)
        d1, d2, d3, d4, d5, d6, d7 = self.model(img)
        pred = d1[:,0,:,:]
        pred = self.normPRED(pred)
        
        # Calculate mean saliency as the score (one value per image)
        scores = self._score_interpolation(pred, method=self.scoring_method)
        
        return {
            'score': scores,
            'saliency_map': pred
        }


class UnifiedAestheticModel:
    def __init__(self, model_type: str = 'u2net', **kwargs):
        """
        Factory class for different saliency and aesthetic models.
        
        Args:
            model_type: Type of model to initialize ('u2net', 'aesthetic', 'drfi', 'basnet')
            **kwargs: Additional arguments to pass to the model constructor
        """
        self.model_type = model_type
        if model_type == 'u2net':
            if 'model_path' not in kwargs:
                kwargs['model_path'] = './models/u2net.pth'
            self.model = U2NetWrapper(**kwargs)
        elif model_type == 'aesthetic':
            from nets import AestheticScorer
            if 'model_path' not in kwargs:
                kwargs['model_path'] = './models/EvaluationNet.pth.tar'
            self.model = AestheticScorer(**kwargs)
        elif model_type == 'drfi':
            from nets import DRFISaliencyModel
            self.model = DRFISaliencyModel(**kwargs)
        elif model_type == 'basnet':
            from nets import BASNetWrapper
            self.model = BASNetWrapper(**kwargs)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def verify_operation(self):
        """Test the model with a random image to verify it's working correctly"""
        test_img = torch.randint(0, 256, (512, 512, 3), dtype=torch.uint8)  # Shape: (H, W, C)
        results = self(test_img)
        smap, scores = results['saliency_map'], results['score']
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Score range: {smap.min():.4f} - {smap.max():.4f}")
        print(f"Mean saliency: {scores[0]:.4f}")
        if isinstance(smap, np.ndarray):
            assert 0 <= smap.min() <= 1, "U^2-Net output out of range!"
            assert 0 <= smap.max() <= 1, "U^2-Net output out of range!"



def saliency_dirs(smap: torch.Tensor) -> np.ndarray:
    """
    Aggregate the dominant directions from a batch of saliency masks.
    
    For each image in the batch, if the sum of activated pixels on the left
    half is greater than on the right, it's counted as "left" (and vice versa).
    Similarly, if the top half has more activation than the bottom, it's counted as "up".
    Empty masks (with all zeros) are ignored.
    
    Args:
        mask_batch: Torch tensor of shape (N, H, W, 3) with identical channels,
                    where 1 indicates activation.
                    
    Returns:
        A numpy array of shape (4,) with aggregated counts in the order:
        [left, right, up, down].
        For example, [2, 0, 1, 1] means two frames are predominantly left,
        one predominantly up, and one predominantly down.
    """
    # Use a single channel since all 3 channels are identical.
    smap = smap[..., 0]  # shape: (N, H, W)
    N, H, W = smap.shape

    # Sum activations on the left/right halves.
    left_sum = smap[:, :, :W // 2].sum(dim=(1, 2))
    right_sum = smap[:, :, W // 2:].sum(dim=(1, 2))
    
    # Sum activations on the top (up) and bottom (down) halves.
    up_sum = smap[:, :H // 2, :].sum(dim=(1, 2))
    down_sum = smap[:, H // 2:, :].sum(dim=(1, 2))
    
    # Identify frames that are non-empty.
    non_empty = smap.sum(dim=(1, 2)) > 0  # Boolean vector, shape (N,)
    non_empty_count = non_empty.sum()

    # Only consider non-empty frames for the directional decision.
    horiz_left = (left_sum > right_sum) & non_empty
    vert_up = (up_sum > down_sum) & non_empty

    left_count = horiz_left.sum().item()
    right_count = (non_empty.sum() - horiz_left.sum()).item()
    up_count = vert_up.sum().item()
    down_count = (non_empty.sum() - vert_up.sum()).item()

    return np.array([left_count, right_count, up_count, down_count])



def saliency_direction(mask_batch):
    """
    Calculate dominant directions from saliency masks.
    
    Args:
        mask_batch: Tensor of shape (N, H, W, 3) with identical channels
                    
    Returns:
        numpy array [left_count, right_count, up_count, down_count]
    """
    # Use a single channel since all 3 are identical
    mask = mask_batch[..., 0]  # shape: (N, H, W)
    N, H, W = mask.shape

    # Calculate directional sums
    left_sum = mask[:, :, :W // 2].sum(dim=(1, 2))
    right_sum = mask[:, :, W // 2:].sum(dim=(1, 2))
    up_sum = mask[:, :H // 2, :].sum(dim=(1, 2))
    down_sum = mask[:, H // 2:, :].sum(dim=(1, 2))
    
    # Identify non-empty frames
    non_empty = mask.sum(dim=(1, 2)) > 0
    
    # Determine directional dominance
    horiz_left = (left_sum > right_sum) & non_empty
    vert_up = (up_sum > down_sum) & non_empty
    
    # Count frames in each direction
    left_count = horiz_left.sum().item()
    right_count = (non_empty.sum() - horiz_left.sum()).item()
    up_count = vert_up.sum().item()
    down_count = (non_empty.sum() - vert_up.sum()).item()

    return np.array([left_count, right_count, up_count, down_count])

class OldClipVQAModel:
    """CLIP-based Video Quality Assessment Model"""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        # Quality text prompts
        self.quality_prompts = [
            "a high quality, sharp video frame with good contrast",
            "a medium quality video frame with some imperfections",
            "a low quality, blurry or noisy video frame"
        ]
        
        # Pre-process text prompts once
        text_inputs = self.processor(
            text=self.quality_prompts,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        with torch.no_grad():
            self.text_features = self.model.get_text_features(**text_inputs)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
            
    def __call__(self, frames):
        """
        Process frames using CLIP-VQA
        
        Args:
            frames: Tensor of shape (B, H, W, 3) or numpy array
            
        Returns:
            dict: Contains 'score' tensor and None for saliency_map
        """
        # Convert frames to PIL images
        frames_pil = []
        
        if isinstance(frames, torch.Tensor):
            # Handle tensor input
            for i in range(frames.shape[0]):
                frame = frames[i].cpu()
                if frame.shape[0] == 3 and len(frame.shape) == 3:  # NCHW format
                    frame = frame.permute(1, 2, 0)
                
                if frame.max() <= 1.0:
                    frame = frame * 255
                
                numpy_frame = frame.numpy().astype(np.uint8)
                frames_pil.append(Image.fromarray(numpy_frame))
        else:
            # Handle numpy array input
            for i in range(frames.shape[0]):
                numpy_frame = frames[i].astype(np.uint8)
                frames_pil.append(Image.fromarray(numpy_frame))
        
        # Process with CLIP model
        prompt_scores = []
        for prompt in self.quality_prompts:
            inputs = self.processor(
                text=[prompt] * len(frames_pil),
                images=frames_pil,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits_per_image.diag()
                prompt_scores.append(scores)
        
        # Take maximum score across prompts
        final_scores = torch.stack(prompt_scores).max(dim=0)[0]
        
        return {
            'score': final_scores.to(self.device),
            'saliency_map': None
        }




class ClipVQAModel:
    """Enhanced CLIP-based Video Quality Assessment Model with Spatiotemporal Aggregation"""

    def __init__(self, device="cuda"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

        # Quality text prompts
        self.quality_prompts = [
            "a high quality, sharp video frame with good contrast",
            "a medium quality video frame with some imperfections",
            "a low quality, blurry or noisy video frame"
        ]

        # Pre-process text prompts once
        text_inputs = self.processor(
            text=self.quality_prompts,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            self.text_features = self.model.get_text_features(**text_inputs)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        # Dynamically retrieve hidden size from the vision encoder configuration
        vision_hidden_size = getattr(self.model.config.vision_config, 'projection_dim', None)
        if vision_hidden_size is None:
            raise ValueError("Unable to retrieve vision hidden size from the model configuration.")
        
        # Spatiotemporal transformer for feature aggregation
        encoder_layer = TransformerEncoderLayer(
            d_model=vision_hidden_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu"
        )
        self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=1).to(device)

        # Regression head for final quality prediction
        self.regression_head = Linear(vision_hidden_size, 1).to(device)

    def __call__(self, frames):
        """
        Args:
            frames: Tensor of shape (B, T, H, W, 3) or numpy array
        
        Returns:
            dict: Contains 'score' tensor.
        """
        # Validate input dimensions
        if isinstance(frames, torch.Tensor):
            if len(frames.shape) != 5 or frames.shape[-1] != 3:
                raise ValueError("Frames should have shape (B, T, H, W, 3).")
        
        elif isinstance(frames, np.ndarray):
            if len(frames.shape) != 5 or frames.shape[-1] != 3:
                raise ValueError("Frames should have shape (B, T, H, W, 3).")
        
        else:
            raise TypeError("Frames must be either a torch.Tensor or numpy.ndarray.")

        # Convert frames to PIL images batch-wise for efficiency
        frames_pil = []
        
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()  # Convert to numpy for PIL processing

        for batch in frames:  # Iterate over batches
            batch_pil = []
            for frame in batch:  # Iterate over temporal frames within a batch
                if frame.max() <= 1.0:  # Scale up if normalized to [0, 1]
                    frame = (frame * 255).astype(np.uint8)
                batch_pil.append(Image.fromarray(frame))
            frames_pil.append(batch_pil)

        # Extract features for all frames in batches
        all_frame_features = []
        
        for batch_pil in frames_pil:
            inputs = self.processor(
                images=batch_pil,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_frame_features.append(image_features)

        # Stack features and pass through temporal transformer
        all_frame_features = torch.stack(all_frame_features, dim=1)  # Shape: (B, T, D)
        
        with torch.no_grad():
            aggregated_features = self.temporal_transformer(all_frame_features)  # Shape: (B, T, D)

        # Pool temporal features and predict quality score
        video_representation = aggregated_features.mean(dim=1)  # Temporal average pooling
        scores = self.regression_head(video_representation).squeeze(-1)  # Final quality score

        return {
            'score': scores.to(self.device)
        }

class ClipIQAModelOld:
    """
    Enhanced CLIP-based Image Quality Assessment Model.
    
    Evaluates image quality using CLIP embeddings compared against quality-related 
    text prompts, following standards from CVPR/ICCV/ICLR research papers.
    """
    
    def __init__(self, 
                 device="cuda", 
                 model_name="openai/clip-vit-large-patch14",
                 use_multi_crop=True,
                 prompt_strategy="graduated"):
        """
        Initialize the enhanced CLIP-IQA model.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            model_name: CLIP model variant. Options:
                - 'openai/clip-vit-base-patch16'
                - 'openai/clip-vit-base-patch32'
                - 'openai/clip-vit-large-patch14' (default, CVPR 2023 recommendation)
                - 'openai/clip-vit-large-patch14-336'
            use_multi_crop: Whether to use multi-crop evaluation (CVPR 2023 standard)
            prompt_strategy: Strategy for text prompts:
                - 'binary': Simple good/bad prompts
                - 'graduated': 5-level quality descriptions (CVPRW 2024)
                - 'multi_attribute': Multiple attribute prompts (ICLR papers)
        """
        self.device = device
        self.model_name = model_name
        self.use_multi_crop = use_multi_crop
        self.prompt_strategy = prompt_strategy
        
        # Initialize CLIP model directly using transformers (not torchmetrics)
        # This gives us more control over the processing pipeline
        self.clip_model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Define prompt strategies
        if prompt_strategy == "binary":
            # Basic binary quality prompts (original implementation)
            self.quality_prompts = [
                "a good quality photo",
                "a bad quality photo"
            ]
            
        elif prompt_strategy == "graduated":
            # 5-level graduated quality prompts (CVPRW 2024)
            self.quality_prompts = [
                "an excellent quality photo with perfect details",
                "a good quality photo with clear details",
                "an average quality photo with acceptable details",
                "a poor quality photo with visible flaws",
                "a terrible quality photo with severe artifacts"
            ]
            # self.quality_prompts = [
            #     "an excellent quality 3D rendered image with perfect details",
            #     "a good quality 3D rendered image with clear details",
            #     "an average quality 3D rendered image with acceptable details",
            #     "a poor quality 3D rendered image with visible flaws",
            #     "a terrible quality 3D rendered image with severe artifacts"
            # ]

            # Corresponding quality weights for aggregation
            self.quality_weights = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0], device=device)
            
        elif prompt_strategy == "multi_attribute":
            # Using attribute-specific assessment (LIQE approach)
            self.use_attributes = True
            
        # Standard attributes for detailed assessment (from Lightning AI docs & academic papers)
        self.attribute_prompts = {
            "brightness": ["a photo with good brightness", "a photo with poor brightness"],
            "sharpness": ["a sharp photo", "a blurry photo"],
            "colorfulness": ["a photo with vibrant colors", "a photo with dull colors"],
            "noisiness": ["a clean photo", "a noisy photo"],
            "contrast": ["a photo with good contrast", "a photo with poor contrast"],
            "natural": ["a natural photo", "a synthetic photo"],
            "beautiful": ["a beautiful photo", "an ugly photo"],
            "realistic": ["a realistic photo", "an unrealistic photo"],
        }
        # self.attribute_prompts = {
        #     "geometry": ["accurate 3D geometry", "distorted 3D geometry"],
        #     "texture": ["realistic surface textures", "blurry or noisy textures"],
        #     "lighting": ["natural lighting and shadows", "unrealistic lighting"],
        #     "view_consistency": ["consistent across viewpoints", "inconsistent across viewpoints"]
        # }

        
        # For multi-crop evaluation
        self.num_crops = 5 if use_multi_crop else 1
    
    def __call__(self, frames):
        """
        Evaluate image quality of frames using CLIP similarity to quality prompts.
        
        Args:
            frames: Tensor of frames in either format:
                   - NCHW format: (B, 3, H, W) with normalized values [-1,1] or [0,1]
                   - NHWC format: (B, H, W, 3) with values typically [0,255]
                   
        Returns:
            dict: Dictionary containing:
                  - 'score': Tensor of quality scores for each frame
                  - 'attribute_scores': (if multi_attribute) Dictionary of attribute scores
                  - 'saliency_map': None (CLIP-IQA doesn't generate saliency maps)
        """
        # Process frames to get PIL images
        pil_images = self._convert_to_pil(frames)
        
        # Compute quality scores based on prompt strategy
        if self.prompt_strategy == "binary":
            quality_scores = self._compute_binary_scores(pil_images)
            
        elif self.prompt_strategy == "graduated":
            quality_scores = self._compute_graduated_scores(pil_images)
            
        elif self.prompt_strategy == "multi_attribute":
            # Multi-attribute assessment (LIQE approach)
            attribute_scores = self.evaluate_attributes(pil_images)
            # Aggregate attribute scores with equal weighting
            quality_scores = sum(attribute_scores.values()) / len(attribute_scores)
            
        # Ensure output is properly shaped for batch processing
        if quality_scores.dim() == 0:
            quality_scores = quality_scores.unsqueeze(0)
            
        result = {
            'score': quality_scores.to(self.device),
            'saliency_map': None
        }
        
        # Add attribute scores if using multi-attribute assessment
        if self.prompt_strategy == "multi_attribute":
            result['attribute_scores'] = attribute_scores
            
        return result
    
    def _convert_to_pil(self, frames):
        """
        Convert various input frame formats to PIL images.
        
        Args:
            frames: Tensor or array of frames
            
        Returns:
            list: List of PIL images
        """
        pil_images = []
        
        # Check input type and format
        if isinstance(frames, torch.Tensor) and frames.dim() == 4:
            for i in range(frames.shape[0]):
                if frames.shape[1] == 3:  # NCHW format
                    # Extract frame and denormalize
                    frame = frames[i].cpu().float()
                    
                    # Denormalize if needed
                    if frame.min() < 0:  # Probably normalized to [-1,1]
                        frame = (frame + 1) / 2  # Convert to [0,1]
                    
                    if frame.max() <= 1.0:  # In [0,1] range, convert to [0,255]
                        frame = frame * 255
                        
                    # Convert to PIL
                    frame_np = frame.permute(1, 2, 0).numpy().astype(np.uint8)
                    pil_images.append(Image.fromarray(frame_np))
                else:  # NHWC format
                    frame = frames[i].cpu()
                    
                    if frame.max() <= 1.0:  # In [0,1] range
                        frame = frame * 255
                        
                    pil_images.append(Image.fromarray(frame.numpy().astype(np.uint8)))
        
        elif isinstance(frames, np.ndarray):
            # Handle numpy array input
            for i in range(frames.shape[0]):
                if frames.shape[3] == 3:  # NHWC format
                    pil_images.append(Image.fromarray(frames[i].astype(np.uint8)))
                else:  # NCHW format
                    frame = frames[i].transpose(1, 2, 0)
                    if frame.max() <= 1.0:
                        frame = frame * 255
                    pil_images.append(Image.fromarray(frame.astype(np.uint8)))
                    
        else:
            raise ValueError(f"Unsupported frame type: {type(frames)}")
            
        return pil_images
    
    def _compute_clip_similarity(self, images, text):
        """
        Compute CLIP similarity between images and text.
        
        Args:
            images: List of PIL images
            text: Text prompt or list of text prompts
            
        Returns:
            torch.Tensor: Similarity scores
        """
        # Process multiple crops if enabled
        if self.use_multi_crop and not getattr(self, 'training', False):
            # For academic standard multi-crop evaluation
            all_scores = []
            
            # For each image, create multiple crops
            for img in images:
                crops = []
                width, height = img.size
                crop_size = min(width, height)
                
                # Center crop
                crops.append(img.resize((DEFAULT_INPUT_SIZE[0], DEFAULT_INPUT_SIZE[1])))
                
                # Corners
                # Top-left
                crops.append(img.crop((0, 0, crop_size, crop_size)).resize(DEFAULT_INPUT_SIZE))
                # Top-right
                crops.append(img.crop((width - crop_size, 0, width, crop_size)).resize(DEFAULT_INPUT_SIZE))
                # Bottom-left
                crops.append(img.crop((0, height - crop_size, crop_size, height)).resize(DEFAULT_INPUT_SIZE))
                # Bottom-right
                crops.append(img.crop((width - crop_size, height - crop_size, width, height)).resize(DEFAULT_INPUT_SIZE))
                
                # Process all crops for this image
                crop_scores = []
                for crop in crops:
                    inputs = self.processor(text=text, images=crop, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.clip_model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        crop_scores.append(logits_per_image)
                
                # Average scores across crops
                img_score = torch.mean(torch.cat(crop_scores), dim=0)
                all_scores.append(img_score)
                
            return torch.cat(all_scores)
            
        else:
            # Standard single-crop processing
            inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                
            return logits_per_image
    
    def _compute_binary_scores(self, images):
        """
        Compute quality scores using binary (good/bad) prompts.
        
        Args:
            images: List of PIL images
            
        Returns:
            torch.Tensor: Quality scores
        """
        # Calculate similarity to good and bad quality prompts
        good_scores = self._compute_clip_similarity(images, self.quality_prompts[0])
        bad_scores = self._compute_clip_similarity(images, self.quality_prompts[1])
        
        # Quality score is the difference between good and bad similarity
        return good_scores - bad_scores
    
    def _compute_graduated_scores(self, images):
        """
        Compute quality scores using graduated prompts (5-level).
        Following CVPRW 2024 recommendations for more fine-grained assessment.
        
        Args:
            images: List of PIL images
            
        Returns:
            torch.Tensor: Aggregated quality scores
        """
        # Calculate similarity to all quality prompts at once
        similarities = self._compute_clip_similarity(images, self.quality_prompts)
        
        # Handle dimension issues - ensure we have a 2D tensor
        if similarities.dim() == 1:
            # If we have a 1D tensor, add batch dimension
            similarities = similarities.unsqueeze(0)
        
        # Only transpose if necessary (if first dim is number of prompts)
        if similarities.size(0) == len(self.quality_prompts):
            similarities = similarities.transpose(0, 1)
        
        # Apply softmax to get probability distribution across quality levels
        probs = torch.nn.functional.softmax(similarities * 10.0, dim=1)  # Scale factor for sharper distribution
        
        # Compute weighted average using quality weights
        weighted_scores = probs * self.quality_weights.unsqueeze(0)
        quality_scores = weighted_scores.sum(dim=1)
        
        # Normalize to 0-1 range
        quality_scores = quality_scores / self.quality_weights.sum()
        
        return quality_scores


    def evaluate_attributes(self, images):
        """
        Perform detailed attribute-level assessment of image quality.
        Based on LIQE approach from CVPR 2023.
        
        Args:
            images: List of PIL images
            
        Returns:
            dict: Dictionary of attribute scores for each frame
        """
        attribute_scores = {}
        
        # Calculate scores for each attribute
        for attr, prompts in self.attribute_prompts.items():
            good_prompt, bad_prompt = prompts
            good_scores = self._compute_clip_similarity(images, good_prompt)
            bad_scores = self._compute_clip_similarity(images, bad_prompt)
            attribute_scores[attr] = good_scores - bad_scores
            
        return attribute_scores


class ClipIQAModel:
    """CLIP-based Image Quality Assessment Model with simplified interface"""
    
    def __init__(self, 
                device="cuda",
                model_name="openai/clip-vit-large-patch14",
                use_multi_crop=True,
                prompt_strategy="multi_attribute",
                crop_size=224,
                crop_overlap=0.5):
        """Initialize CLIP-IQA model with core parameters"""
        print(f"Initializing ClipIQAModel with following configuration:")
        
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        print(f"  - Using device: {self.device}")
        
        self.use_multi_crop = use_multi_crop
        print(f"  - Multi-crop enabled: {self.use_multi_crop}")
        
        self.crop_size = crop_size
        print(f"  - Crop size: {self.crop_size}px")
        
        self.crop_overlap = max(0.0, min(crop_overlap, 0.95))
        print(f"  - Crop overlap: {self.crop_overlap:.2f} ({int(self.crop_overlap*100)}%)")
        
        try:
            # Load CLIP model
            self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            print(f"  - CLIP model loaded successfully")
        except Exception as e:
            print(f"  - ERROR loading CLIP model: {str(e)}")
            raise RuntimeError(f"Failed to load CLIP model '{model_name}': {str(e)}")
        
        print(f"  - Setting up prompts with strategy: {prompt_strategy}")
        self._setup_prompts(prompt_strategy)
        print(f"  - Configured {len(self.quality_prompts)} attribute(s) for assessment:")
        
    def _setup_prompts(self, strategy):
        """Configure quality prompts based on strategy"""
        prompt_configs = {
            'binary': {
                'quality': ("High quality photo", "Low quality photo"),
            },
            'graduated': {
                'quality': ("Very high quality photo", "High quality photo", 
                           "Medium quality photo", "Low quality photo", 
                           "Very low quality photo"),
            },
            'multi_attribute': {
                'geometric_integrity': ("Precise 3D geometry with sharp edges", 
                                    "Blurry 3D shapes with floating points"),
                'depth_consistency': ("Coherent depth layers", 
                                    "Floating debris artifacts"),
                'splat_quality': ("Well-defined Gaussian splats",
                                "Overlapping/Underdefined splats")
            }

        }
        
        if strategy not in prompt_configs:
            strategy = 'binary'  # Default to binary if invalid
            
        self.quality_prompts = prompt_configs[strategy]
        self.is_graduated = (strategy == 'graduated')
        
    def __call__(self, images):
        """Process images to assess their quality"""
        # Convert to list of PIL images
        pil_images = self._to_pil_images(images)
        
        if not pil_images:
            return {'score': torch.tensor([], device=self.device)}
            
        # Use multi-crop if enabled and images are large enough
        if self.use_multi_crop and any(img.width > self.crop_size or img.height > self.crop_size 
                                     for img in pil_images):
            return self._assess_multi_crop(pil_images)
        else:
            return self._assess_direct(pil_images)

    def _to_pil_images(self, images):
        """Convert various input formats to list of PIL images"""
        # Handle numpy array input (most common case)
        if isinstance(images, np.ndarray):
            # For batch of images (B, H, W, C)
            if images.ndim == 4 and images.shape[3] in (1, 3, 4):
                return [Image.fromarray(img) for img in images]
            
            # For single image (H, W, C), wrap in a list first
            if images.ndim == 3 and images.shape[2] in (1, 3, 4):
                return [Image.fromarray(images)]
                
            raise ValueError(f"Unsupported array shape: {images.shape}")
        
        # Handle tensor input
        if isinstance(images, torch.Tensor):
            # Convert to numpy array
            images_np = images.cpu().numpy()
            
            # Handle NCHW format (B, C, H, W)
            if images.ndim == 4 and images.shape[1] in (1, 3):
                images_np = images_np.transpose(0, 2, 3, 1)
            
            # Handle CHW format (C, H, W)
            elif images.ndim == 3 and images.shape[0] in (1, 3):
                images_np = images_np.transpose(1, 2, 0)
            
            # Ensure uint8 format
            if images_np.max() <= 1.0:
                images_np = (images_np * 255).astype(np.uint8)
            else:
                images_np = images_np.astype(np.uint8)
            
            # Recursive call with numpy array
            return self._to_pil_images(images_np)
        
        # Handle list or PIL image
        if isinstance(images, list):
            return [img if isinstance(img, Image.Image) else Image.fromarray(np.array(img)) for img in images]
        if isinstance(images, Image.Image):
            return [images]
        
        # Last resort conversion
        try:
            return [Image.fromarray(np.array(images))]
        except:
            raise TypeError(f"Cannot convert {type(images)} to PIL Image")


    def _assess_direct(self, images):
        """Direct assessment without cropping"""
        # Prepare prompts
        prompts = []
        for attr_prompts in self.quality_prompts.values():
            prompts.extend(attr_prompts)
            
        # Process through CLIP
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)
        
        # Calculate scores
        scores = self._compute_scores(probs)
        
        return scores

    def _assess_multi_crop(self, images):
        """Multi-crop assessment with overlapping windows"""
        all_scores = []
        all_attr_scores = {attr: [] for attr in self.quality_prompts.keys()}
        
        for img in images:
            # Generate crops
            crops = self._generate_crops(img)
            
            # Process crops through direct assessment
            crop_scores = self._assess_direct(crops)
            
            # Average the score
            avg_score = torch.mean(crop_scores['score'])
            all_scores.append(avg_score)
            
            # Average the attribute scores
            for attr, scores in crop_scores['attribute_scores'].items():
                all_attr_scores[attr].append(torch.mean(scores))
        
        # Stack results if we have any
        if all_scores:
            return {
                'score': torch.stack(all_scores),
                'attribute_scores': {
                    attr: torch.stack(scores) 
                    for attr, scores in all_attr_scores.items()
                }
            }
        else:
            return {
                'score': torch.tensor([], device=self.device),
                'attribute_scores': {
                    attr: torch.tensor([], device=self.device)
                    for attr in self.quality_prompts.keys()
                }
            }

    def _generate_crops(self, image):
        """Generate overlapping crops"""
        # Convert to numpy if PIL
        if isinstance(image, Image.Image):
            width, height = image.size
            img_np = np.array(image)
        else:
            height, width = image.shape[:2]
            img_np = image
            
        # Handle small images
        if height < self.crop_size or width < self.crop_size:
            return [image]
        
        # Calculate stride with overlap
        stride = int(self.crop_size * (1 - self.crop_overlap))
        crops = []
        
        # Generate crops
        for y in range(0, height - self.crop_size + 1, stride):
            for x in range(0, width - self.crop_size + 1, stride):
                crop = img_np[y:y + self.crop_size, x:x + self.crop_size]
                crops.append(Image.fromarray(crop))
                
        # Add center crop if not already included
        center_y = (height - self.crop_size) // 2
        center_x = (width - self.crop_size) // 2
        center_included = any(y <= center_y < y + stride and x <= center_x < x + stride 
                             for y in range(0, height - self.crop_size + 1, stride)
                             for x in range(0, width - self.crop_size + 1, stride))
        
        if not center_included:
            center_crop = img_np[center_y:center_y + self.crop_size, center_x:center_x + self.crop_size]
            crops.insert(0, Image.fromarray(center_crop))
            
        return crops
    
    def _compute_scores(self, probs):
        """Calculate quality scores from CLIP probabilities"""
        batch_size = probs.shape[0]
        scores = torch.zeros(batch_size, device=self.device)
        attribute_scores = {}
        
        if self.is_graduated:
            # For graduated scale
            prompt_len = len(next(iter(self.quality_prompts.values())))
            weights = torch.linspace(1.0, 0.0, prompt_len, device=self.device)
            scores = torch.sum(probs * weights.unsqueeze(0), dim=1)
            attribute_scores = {'quality': scores}
        else:
            # For binary or multi-attribute
            idx = 0
            for attr, prompts in self.quality_prompts.items():
                if len(prompts) == 2:  # Binary pair (positive/negative)
                    attr_score = probs[:, idx] - probs[:, idx+1]
                    attribute_scores[attr] = attr_score
                    idx += 2
                else:
                    # For multiple prompts
                    attr_len = len(prompts)
                    attr_probs = probs[:, idx:idx+attr_len]
                    weights = torch.linspace(1.0, 0.0, attr_len, device=self.device)
                    attr_score = torch.sum(attr_probs * weights.unsqueeze(0), dim=1)
                    attribute_scores[attr] = attr_score
                    idx += attr_len
            
            # Overall score is mean of attribute scores
            scores = torch.stack(list(attribute_scores.values()), dim=1).mean(dim=1)
            
        return {
            'score': scores,
            'attribute_scores': attribute_scores
        }
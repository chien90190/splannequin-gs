# assessment/models.py
import numpy as np
import torch
import pyiqa
from typing import Dict, Any
from nets import UnifiedAestheticModel, ClipVQAModel

def _log(v, m): 
    if v: 
        print(m)

class BaseModel:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    def predict(self, images: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError

class AestheticModelWrapper(BaseModel):
    def __init__(self, device: str, verbose: bool = False):
        super().__init__(verbose)
        _log(self.verbose, f"[Aesthetic] init on {device}")
        self.model = UnifiedAestheticModel(model_type="aesthetic", model_path="./models/EvaluationNet.pth.tar", device=device).model
        self.device = device
    def __call__(self, images: torch.Tensor) -> Dict[str, Any]:
        return self.model(images)

class U2NetModelWrapper(BaseModel):
    def __init__(self, device: str, verbose: bool = False):
        super().__init__(verbose)
        _log(self.verbose, f"[U2Net] init on {device}")
        self.model = UnifiedAestheticModel(model_type="u2net", model_path="./models/u2net.pth", device=device).model
        self.device = device
    def __call__(self, images: torch.tensor) -> Dict[str, Any]:
        t = torch.from_numpy(images).to(self.device)
        with torch.no_grad():
            return self.model(images)

class ClipVQAModelWrapper(BaseModel):
    def __init__(self, device: str, verbose: bool = False):
        super().__init__(verbose)
        _log(self.verbose, f"[Clip-VQA] init on {device}")
        self.model = ClipVQAModel(device=device)
        self.device = device
    def predict(self, images: np.ndarray) -> Dict[str, Any]:
        t = torch.from_numpy(images).to(self.device)
        with torch.no_grad():
            return self.model(t)

def create_model(model_type: str, device: str, clip_model: str, clip_prompt_strategy: str, verbose: bool):
    if model_type == "aesthetic":
        return AestheticModelWrapper(device, verbose)
    elif model_type == "u2net":
        return U2NetModelWrapper(device, verbose)
    elif model_type == "clip-vqa":
        return ClipVQAModelWrapper(device, verbose)
    elif model_type in ["topiq_nr", "musiq-spaq", "musiq", "hyperiqa", "pi", "liqe", "clipiqa"]:
        return pyiqa.create_metric(model_type, device=device)
    raise ValueError(f"Unknown model_type: {model_type}")

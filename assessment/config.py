# assessment/config.py
from dataclasses import dataclass

@dataclass
class Config:
    model_type: str = "aesthetic"
    clip_model: str = "openai/clip-vit-large-patch14"
    clip_prompt_strategy: str = "graduated"
    batch_size: int = 16
    num_workers: int = 4
    smap_filter: float = 0.5

    save_details: bool = False
    save_summary: bool = False
    save_plots: bool = False
    save_visualizations: bool = False

    mode: str = "normal"
    verbose: bool = False  # NEW

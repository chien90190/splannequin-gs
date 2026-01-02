# assessment/assess.py
import csv
import os, gc, torch
import numpy as np
from typing import Dict, List
from tqdm import tqdm
from .config import Config
from .models import create_model
from .loader import load_batch
from .process import compute_sequence_metrics, format_output
from .discovery import find_image_directories, list_images
from .writers import write_detailed_csv, write_summary_json, write_quality_plot

def _log(verbose, msg): 
    if verbose: print(msg)

class QualityAssessor:
    def __init__(self, config: Config):
        self.cfg = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = create_model(self.cfg.model_type, 
                                  self.device, 
                                  self.cfg.clip_model, 
                                  self.cfg.clip_prompt_strategy, 
                                  self.cfg.verbose
                                  )
        _log(self.cfg.verbose, f"[Init] {self.cfg.model_type}")

    def preprocessing(self, batch_files):
        images, indices, failed = load_batch(batch_files, self.cfg.num_workers)
        if failed:
            return None, None, failed
        
        images = torch.tensor(np.stack(images, axis=0)).to(self.device)

        if self.cfg.model_type in ["topiq_nr", "musiq-spaq", "musiq", "hyperiqa", "pi", "liqe", "clipiqa"]:
            if images.dtype != torch.float32 or images.max() > 1.0:
                images = images.float() / 255.0
            return images.permute(0, 3, 1, 2).contiguous(), indices, failed

        return images, indices, failed

    def quality_assessment(self, sequence_path: str) -> Dict:
        """Process all images in one sequence folder"""
        files = list_images(sequence_path, verbose=self.cfg.verbose)
        seq_name = os.path.basename(sequence_path)
        
        all_scores = []
        frame_indices = []
        attr_scores = {}
        
        with torch.no_grad():
            with tqdm(total=len(files), desc=f"seq:{seq_name}", leave=False, disable=not self.cfg.verbose) as pbar:
                for i in range(0, len(files), self.cfg.batch_size):

                    # images are 0-1 torch.cuda.FloatTensor
                    batch_files = files[i:i + self.cfg.batch_size]
                    images, indices, failed = self.preprocessing(batch_files)
                    if failed:
                        pbar.update(len(batch_files))
                        continue
                    
                    raw_output = self.model(images)
                    output = format_output(raw_output)

                    result = {"frame_scores": output["score"], "batch_sum": sum(output["score"])}                
                    if "attribute_scores" in output: 
                        for attr, values in output["attribute_scores"].items():
                            attr_scores.setdefault(attr, []).extend(values)

                    if "saliency_map" in output: 
                        result["saliency_map"] = output["saliency_map"]
                    
                    # Collect results
                    all_scores.extend(result["frame_scores"])
                    frame_indices.extend(indices[:len(result["frame_scores"])])
                    pbar.update(len(images))

        # sort
        if all_scores:
            sort_map = sorted(range(len(frame_indices)), key=lambda i: frame_indices[i])
            frame_indices = [frame_indices[i] for i in sort_map]
            all_scores = [all_scores[i] for i in sort_map]

            if attr_scores:
                for attr in attr_scores:
                    attr_scores[attr] = [attr_scores[attr][i] for i in sort_map]
        
        result = {
            "sequence_path": sequence_path,
            "sequence_name": seq_name,
            "metric_type": self.cfg.model_type,
            "frame_scores": all_scores,
            "frame_indices": frame_indices,
        }
        if attr_scores:
            result["attribute_scores"] = attr_scores
        return result

    def assess_directory(self, dir_path: str, sequences: list) -> Dict:
        """
        Process all sequences in a directory
        """
        results = {}
        with tqdm(total=len(sequences), desc=f"[{self.scene}]sequences", leave=False, disable=not self.cfg.verbose) as pbar:
            for seq_path in sequences:
                seq_result = self.quality_assessment(seq_path)
                seq_stats = compute_sequence_metrics(seq_result['frame_scores'], seq_result['frame_indices'], verbose=self.cfg.verbose)
                merged_seq = {**seq_result, **seq_stats}
                results[merged_seq["sequence_name"]] = merged_seq
                self._save_results(seq_result["sequence_name"], seq_result, self.out_dir)
                pbar.update(1)
        
        return {"directory": dir_path,
                "sequences": results,
                }

    def run(self, base_dir: str, mode: str = "normal", type_name: str = "", out_dir: str = "./results"):
        
        # find path
        self.method = base_dir.split('/')[2]
        self.out_dir = os.path.join(out_dir, self.method, self.cfg.model_type)
        pairs = find_image_directories(base_dir, mode, type_name, verbose=self.cfg.verbose, return_sequences=True)
        all_results = {}
        
        for directory, sequences in pairs:
            self.scene = directory.split('/')[4]
            _log(self.cfg.verbose, f"\n{'='*100}\n[Running] {directory} with {len(sequences)} sequences.")
            
            # Process directory
            dir_result = self.assess_directory(directory, sequences)
            all_results[directory] = dir_result
            
            # Save outputs
            if self.cfg.save_summary:
                path = os.path.join(self.out_dir, f"{self.scene}_{self.cfg.model_type}_summary.json")
                write_summary_json(dir_result, path, verbose=self.cfg.verbose)
            
            torch.cuda.empty_cache()
            gc.collect()
        
        return all_results
    
    def _save_results(self, seq_name, seq_result, out_dir: str):
        """Save results in requested formats"""
        if self.cfg.save_details:
            path = os.path.join(out_dir, self.scene, f"{seq_name}_{self.cfg.model_type}_details.csv")
            write_detailed_csv(seq_result, path, verbose=self.cfg.verbose)
        
        if self.cfg.save_plots:
            plot_dir = os.path.join(out_dir, self.scene, "plots")
            write_quality_plot(seq_result, plot_dir, self.cfg.model_type, verbose=self.cfg.verbose)
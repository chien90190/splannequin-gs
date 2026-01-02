# assessment/cli.py
import argparse
from .config import Config
from .assess import QualityAssessor

def main():
    p = argparse.ArgumentParser(description="Assess image sequences with referenceless IQA/Aesthetics/VQA.")
    p.add_argument("input_path", help="Base directory path")
    p.add_argument("--output_folder", default="./results/referenceless")
    p.add_argument("--model_type", default="aesthetic", choices=['aesthetic','clip-iqa','clip-vqa','u2net'])
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--clip_prompt_strategy", default="graduated", choices=['binary','graduated','multi_attribute'])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--mode", default="normal")
    p.add_argument("--type", default="")
    p.add_argument("--save_summary", action="store_true")
    p.add_argument("--save_details", action="store_true")
    p.add_argument("--save_plots", action="store_true")
    args = p.parse_args()

    cfg = Config(
        model_type=args.model_type,
        clip_model=args.clip_model,
        clip_prompt_strategy=args.clip_prompt_strategy,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_summary=args.save_summary,
        save_details=args.save_details,
        save_plots=args.save_plots,
        mode=args.mode,
    )
    qa = QualityAssessor(cfg)
    qa.run(args.input_path, mode=args.mode, type_name=args.type, out_dir=args.output_folder)

if __name__ == "__main__":
    main()

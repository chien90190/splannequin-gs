import os
import argparse
from pathlib import Path
from assessment import QualityAssessor, Config


def main(args):
    args.out_dir += f'/{args.render_mode}'

    config = Config(
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_summary=args.save_summary,
        save_details=args.save_details,
        save_plots=args.save_plots,
    )

    assessor = QualityAssessor(config)
    all_results = {}
    for base_dir in args.base_dir:
        print(f"[INFO] Processing directory: {base_dir}")
        results = assessor.run(base_dir,
                               mode=args.render_mode,
                               out_dir=str(args.out_dir))
        if results:
            all_results.update(results)

    if not all_results:
        print("[INFO] No directories found or no sequences to process.")
        return
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quality assessment")
    parser.add_argument("base_dir", type=str, nargs='+', help="One or more base directories to process")
    parser.add_argument("--out_dir", type=str, default='./results', help="Output directory for results")
    parser.add_argument("--model_type", type=str, default="musiq", choices=["aesthetic", "clipiqa", "musiq", "musiq-spaq", "topiq_nr", "hyperiqa", "pi", "liqe"], help="Model type to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--render_mode", type=str, default="fixed-time-view", choices=["fixed-time-view", "time-sweeping-view"], help="Processing mode")
    parser.add_argument("--save_summary", action="store_true", help="Save summary results")
    parser.add_argument("--save_details", action="store_true", help="Save detailed results")
    parser.add_argument("--save_plots", action="store_true", help="Save plots")
    args = parser.parse_args()
    main(args)

import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run batch inference on multiple checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing checkpoints (e.g., checkpoints/stage2_braid)")
    parser.add_argument("--target", type=str, required=True, help="Target background image path")
    parser.add_argument("--sketch", type=str, required=True, help="Source sketch path")
    parser.add_argument("--matte", type=str, required=True, help="Source matte path")
    parser.add_argument("--output_dir", type=str, default="results/batch_inference", help="Output directory")
    
    # Inference Args
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--x", type=int, default=0)
    parser.add_argument("--y", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--bg_start_ratio", type=float, default=0.5)
    parser.add_argument("--checkpoints", type=str, nargs="+", help="Optional: List of specific checkpoints to test (e.g., stage2_checkpoint-30)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Checkpoints to test
    if args.checkpoints:
        checkpoints = args.checkpoints
    else:
        checkpoints = [
            "stage2_checkpoint-5",
            "stage2_checkpoint-10",
            "stage2_checkpoint-15",
            "stage2_checkpoint-20",
            "stage2_checkpoint-25",
            "stage2_checkpoint-30"
        ]
    
    for ckpt_name in checkpoints:
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        
        if not os.path.exists(ckpt_path):
            print(f"Skipping {ckpt_name} (Not found)")
            continue
            
        print(f"Running inference for {ckpt_name}...")
        
        output_filename = f"result_{ckpt_name}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        
        cmd = [
            "python", "inference_sd3_5_masked.py",
            "--checkpoint_dir", ckpt_path,
            "--image_path", args.target,
            "--mask_path", args.matte,
            "--sketch_path", args.sketch,
            "--output_path", output_path,
            "--num_inference_steps", str(args.steps),
            "--guidance_scale", str(args.guidance),
            "--bg_start_ratio", str(args.bg_start_ratio),
            "--x", str(args.x),
            "--y", str(args.y),
            "--scale", str(args.scale),
            "--seed", str(args.seed)
        ]
        
        # Run command
        subprocess.run(cmd)
        
    print(f"Batch inference complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()

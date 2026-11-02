"""
Automated Pipeline Orchestrator for Construction Site AI.
Sequentially runs:
1. Conditional Diffusion (100 epochs, resume from checkpoint)
2. VQ-VAE & ImageGPT Transformer (100 epochs)
3. Unified Cross-Model Benchmark Evaluation (N=2,048 samples/model)
"""

import os
import sys
import time
import subprocess
import shutil

def run_step(step_name, cmd_args):
    print(f"\n{'='*70}")
    print(f"PIPELINE STEP: {step_name.upper()}")
    print(f"Command: {' '.join(cmd_args)}")
    print(f"{'='*70}\n")
    t0 = time.time()
    result = subprocess.run(cmd_args, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with return code {result.returncode} after {elapsed:.1f}s.")
        sys.exit(result.returncode)
    print(f"\n[SUCCESS] Step '{step_name}' completed in {elapsed:.1f}s ({elapsed/60:.2f} min).\n")
    return elapsed


def main():
    t_start = time.time()
    print("\n" + "#"*70)
    print("STARTING AUTONOMOUS REMAINING PHASES (DIFFUSION + TRANSFORMER + EVAL)")
    print("#"*70)

    # Step 1: Conditional Diffusion (100 Epochs)
    if not (os.path.exists("outputs/synthetic/diffusion/manifest.csv") and os.path.exists("runs/diffusion_baseline/best.pt")):
        run_step(
            "Conditional Diffusion Training (100 Epochs)",
            [sys.executable, "-m", "src.train", "--config", "configs/diffusion.yaml"]
        )
    else:
        print("\n[INFO] Conditional Diffusion (100 Epochs) is already completed. Skipping to Step 2.\n")

    # Step 2: VQ-VAE & Transformer (100 Epochs)
    run_step(
        "VQ-VAE & Transformer Training (100 Epochs)",
        [sys.executable, "-m", "src.train", "--config", "configs/vqvae_transformer.yaml"]
    )

    # Step 3: Unified Cross-Model Benchmark
    run_step(
        "Cross-Model Benchmark Evaluation",
        [sys.executable, "-m", "src.eval_benchmark"]
    )

    total_time = time.time() - t_start
    print("\n" + "#"*70)
    print(f"ALL GENERATIVE PHASES & BENCHMARKS COMPLETE IN {total_time:.1f}s ({total_time/60:.2f} min)!")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()

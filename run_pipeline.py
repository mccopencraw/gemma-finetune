"""
Quick Start Script - Runs Complete Pipeline
Usage: python run_pipeline.py --step all --api_key YOUR_KEY
"""

import argparse
import subprocess
import sys
import os

def run_step(step_name, script_path):
    """Run a pipeline step"""
    print(f"\n{'='*60}")
    print(f"Running: {step_name}")
    print(f"{'='*60}")
    
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ {step_name} failed!")
        return False
    
    print(f"✓ {step_name} completed")
    return True

def main():
    parser = argparse.ArgumentParser(description="Gemma Fine-tuning Pipeline")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "collect", "label", "train", "evaluate"],
                        help="Which step to run")
    parser.add_argument("--api_key", type=str, help="DeepSeek API key")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    args = parser.parse_args()
    
    print("="*60)
    print("Gemma 3 Fine-tuning Pipeline")
    print("="*60)
    
    # Set API key if provided
    if args.api_key:
        os.environ["DEEPSEEK_API_KEY"] = args.api_key
        # Update CONFIG.py
        with open("CONFIG.py", "r") as f:
            config = f.read()
        config = config.replace('DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY_HERE"',
                                f'DEEPSEEK_API_KEY = "{args.api_key}"')
        with open("CONFIG.py", "w") as f:
            f.write(config)
    
    # Update epochs if specified
    if args.epochs != 3:
        with open("CONFIG.py", "r") as f:
            config = f.read()
        config = config.replace("TRAIN_EPOCHS = 3", f"TRAIN_EPOCHS = {args.epochs}")
        with open("CONFIG.py", "w") as f:
            f.write(config)
    
    # Run steps
    success = True
    
    if args.step in ["all", "collect"]:
        if not run_step("Data Collection", "data_collection/collect_data.py"):
            success = False
    
    if args.step in ["all", "label"]:
        if not run_step("Data Labeling", "labeling/label_data.py"):
            success = False
    
    if args.step in ["all", "train"]:
        if not run_step("Training", "training/train.py"):
            success = False
    
    if args.step in ["all", "evaluate"]:
        if not run_step("Evaluation", "training/evaluate.py"):
            success = False
    
    if success:
        print("\n" + "="*60)
        print("✅ Pipeline completed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Pipeline failed!")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()

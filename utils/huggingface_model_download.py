"""
Utility script to pre-download model weights and configuration from HuggingFace.
This allows any supported model to be used in offline mode.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Set environment variable to disable NumPy 2.x compatibility warnings
os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

# Check if running in conda environment
if 'CONDA_PREFIX' not in os.environ:
    print("WARNING: Not running in a conda environment.")
    print("Please activate the internvl_env conda environment first:")
    print("conda activate internvl_env")
    sys.exit(1)

def download_model(model_name, output_dir=None):
    """
    Download model weights, configuration, and processor for offline use.

    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save a copy of the cached files (optional)
    """
    print(f"Downloading {model_name}...")

    # Create output directory if it doesn't exist
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Try a simpler download approach using git clone
    git_url = f"https://huggingface.co/{model_name}"
    
    # Use git to download the model
    if output_dir:
        cmd = f"git lfs install && git clone {git_url} {output_dir}"
    else:
        # Set default location in cache
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'models', model_name.replace('/', '_'))
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cmd = f"git lfs install && git clone {git_url} {cache_dir}"
        output_dir = cache_dir
    
    print(f"Running: {cmd}")
    
    try:
        subprocess.check_call(cmd, shell=True)
        print(f"Model {model_name} downloaded successfully to {output_dir}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error downloading model: {e}")
        print("This could be due to missing git-lfs. Try installing it with:")
        print("  conda install -c conda-forge git-lfs")
        print("  git lfs install")
        sys.exit(1)

    return output_dir

def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace model for offline use"
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="HuggingFace model name to download"
    )
    parser.add_argument(
        "--output_dir", 
        help="Optional directory to save model files"
    )

    args = parser.parse_args()
    
    output_dir = download_model(args.model_name, args.output_dir)

    # Print instructions for offline use
    print("\nTo use the downloaded model in offline mode:")
    print(f"Update config.yaml to set pretrained_path: \"{output_dir}\"")
    print(f"Then run: python main.py --config config/config.yaml --mode train")

if __name__ == "__main__":
    main()
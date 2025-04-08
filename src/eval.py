import argparse
import yaml
import os
from pathlib import Path
import subprocess
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def load_model_info(checkpoint_path, cfg, use_base_model=False):
    """Prepare model information for evaluation"""
    temp_model_path = Path("temp_model_for_eval")
    temp_model_path.mkdir(exist_ok=True)
    
    # First, load the base HF model to see its structure
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg['model']['name'],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    print("\nBase HF model structure:")
    for name, _ in base_model.named_parameters():
        print(name)
    
    # Now continue with your model loading
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    print("Model configuration:", cfg['model']['name'])
    
    # Create model configuration with reduced size if specified
    model_config = AutoConfig.from_pretrained(cfg['model']['name'])
    if cfg['model']['student'].get('reduce_size', False):
        factor = cfg['model']['student']['size_reduction_factor']
        print(f"Reducing model size by factor of {factor}")
        model_config.hidden_size = model_config.hidden_size // factor
        model_config.intermediate_size = model_config.intermediate_size // factor
        if hasattr(model_config, 'num_attention_heads'):
            model_config.num_attention_heads = model_config.num_attention_heads // factor
    
    # Load your model
    model = AutoModelForCausalLM.from_pretrained(
        cfg['model']['name'],
        config=model_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("\nYour model structure:")
    for name, _ in model.named_parameters():
        print(name)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_dict = checkpoint['student_model_state_dict']
    
    print("\nCheckpoint keys:")
    for key in checkpoint_dict.keys():
        print(key)
    
    # Handle weight tying for LLaMA
    if 'lm_head.weight' in checkpoint_dict:
        print("\nHandling weight tying...")
        # For LLaMA, we need to keep both lm_head.weight and embed_tokens.weight identical
        print("Ensuring embed_tokens and lm_head weights are identical")
        checkpoint_dict['model.embed_tokens.weight'] = checkpoint_dict['lm_head.weight'].clone()
        # Don't delete lm_head.weight as the model expects it to exist
    
    try:
        model.load_state_dict(checkpoint_dict, strict=True)
    except Exception as e:
        print(f"\nError loading state dict: {e}")
        raise
    
    # Save the model
    model.save_pretrained(temp_model_path)
    
    # Copy tokenizer files from the base model
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'])
    tokenizer.save_pretrained(temp_model_path)
    
    return str(temp_model_path.absolute())

def run_mmlu_eval(model_path, output_dir):
    """Run MMLU evaluation using lm-evaluation-harness"""
    output_file = Path(output_dir) / "mmlu_results.json"
    
    # Construct the command
    cmd = [
        "accelerate", "launch", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
        "--tasks", "mmlu",
        "--num_fewshot", "5",
        "--device", "cuda:0",
        "--batch_size", "auto",
        "--output_path", str(output_file)
    ]
    
    print("\nRunning evaluation command:")
    print(" ".join(cmd))
    
    try:
        # Run the evaluation
        subprocess.run(cmd, check=True)
        
        # Check if the file exists and is a file (not a directory)
        if output_file.is_file():
            with open(output_file, 'r') as f:
                results = json.load(f)
            return results
        else:
            print(f"Warning: Results file not found at {output_file}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with error: {e}")
        return None
    finally:
        # Cleanup temporary model directory
        if Path("temp_model_for_eval").exists():
            import shutil
            shutil.rmtree("temp_model_for_eval")

def main():
    parser = argparse.ArgumentParser(description='Run MMLU evaluation on a trained model')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation results')
    parser.add_argument('--use_base_model', action='store_true', help='Use base HF model instead of checkpoint')
    args = parser.parse_args()
    
    if not args.use_base_model and not args.checkpoint:
        parser.error("Either --checkpoint or --use_base_model must be provided")
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get model info and run evaluation
    model_path = load_model_info(args.checkpoint, cfg, args.use_base_model)
    results = run_mmlu_eval(model_path, args.output_dir)
    
    if results:
        print("\nMMLU Evaluation Results:")
        if 'results' in results:
            for task, metrics in results['results'].items():
                if 'acc' in metrics:
                    print(f"{task}: {metrics['acc']*100:.2f}%")
        else:
            print("Results format unexpected. Check the output file for details.")

if __name__ == "__main__":
    main()
import argparse
import yaml
import os
from pathlib import Path
import subprocess
import json
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def load_model_info(checkpoint_path, cfg, use_base_model=False, use_teacher_model=False):
    """Prepare model information for evaluation - FIXED VERSION"""
    temp_model_path = Path("temp_model_for_eval")
    temp_model_path.mkdir(exist_ok=True)
    
    if use_teacher_model:
        # Load teacher model as full base model (no quantization)
        print("Loading teacher model (full-size base model, no quantization)")
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model']['name'],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model.eval()
        print("Teacher model (full base) loaded successfully")
        
    elif use_base_model:
        # Just use base model with student configuration
        model_config = AutoConfig.from_pretrained(cfg['model']['name'])
        if cfg['model']['student'].get('reduce_size', False):
            factor = cfg['model']['student']['size_reduction_factor']
            print(f"Using base model with reduced size (factor: {factor})")
            # Match training: reduce num_hidden_layers and intermediate_size only
            model_config.num_hidden_layers = max(1, model_config.num_hidden_layers // factor)
            model_config.intermediate_size = max(1, model_config.intermediate_size // factor)
        
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model']['name'],
            config=model_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("Using base model with student configuration")
        
    else:
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        
        # Create model configuration (match training config exactly)
        model_config = AutoConfig.from_pretrained(cfg['model']['name'])
        if cfg['model']['student'].get('reduce_size', False):
            factor = cfg['model']['student']['size_reduction_factor']
            print(f"Reducing model size by factor of {factor}")
            # Match training: reduce num_hidden_layers and intermediate_size only
            model_config.num_hidden_layers = max(1, model_config.num_hidden_layers // factor)
            model_config.intermediate_size = max(1, model_config.intermediate_size // factor)
        
        # Load model with correct config
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model']['name'],
            config=model_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'student_model_state_dict' not in checkpoint:
            raise ValueError("Checkpoint missing 'student_model_state_dict'")
        
        state_dict = checkpoint['student_model_state_dict']
        print(f"Checkpoint contains {len(state_dict)} parameters")
        
        # CRITICAL FIX: DON'T overwrite embeddings!
        # Just load the state dict as-is
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        
        print("Model loaded successfully without corrupting embeddings")
    
    # Save the model
    model.save_pretrained(temp_model_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'])
    tokenizer.save_pretrained(temp_model_path)
    
    return str(temp_model_path.absolute())

def create_model_identifier(cfg, checkpoint_path, use_base_model, use_teacher_model, medical_only):
    """Create a descriptive identifier for the model being evaluated"""
    # Extract base model name (remove path separators)
    base_model = cfg['model']['name'].replace('/', '_').replace('-', '_')
    
    # Add model type info
    if use_teacher_model:
        model_type = "teacher"
    elif use_base_model:
        model_type = "base"
    else:
        # Extract checkpoint name from path
        checkpoint_name = Path(checkpoint_path).stem if checkpoint_path else "unknown"
        model_type = f"checkpoint_{checkpoint_name}"
    
    # Add evaluation scope
    eval_scope = "medical" if medical_only else "full"
    
    # Add size reduction info if applicable (not for teacher model)
    size_info = ""
    if not use_teacher_model and cfg['model']['student'].get('reduce_size', False):
        factor = cfg['model']['student']['size_reduction_factor']
        size_info = f"_reduced_{factor}x"
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{base_model}_{model_type}{size_info}_{eval_scope}_{timestamp}"

def run_mmlu_eval(model_path, output_dir, cfg, checkpoint_path, use_base_model, use_teacher_model, medical_only=False):
    """Run MMLU evaluation using lm-evaluation-harness"""
    # Create descriptive filename
    model_id = create_model_identifier(cfg, checkpoint_path, use_base_model, use_teacher_model, medical_only)
    output_file = Path(output_dir) / f"mmlu_results_{model_id}.json"
    
    print(f"Results will be saved as: {output_file}")
    
    # Select tasks based on medical_only flag
    if medical_only:
        tasks = "mmlu_anatomy,mmlu_clinical_knowledge,mmlu_college_medicine,mmlu_medical_genetics,mmlu_professional_medicine"
        print("Running evaluation on medical MMLU tasks only")
    else:
        tasks = "mmlu"
        print("Running evaluation on all MMLU tasks")
    
    # Construct the command
    cmd = [
        "accelerate", "launch", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
        "--tasks", tasks,
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
            
            # Add model metadata to results
            if 'config' not in results:
                results['config'] = {}
            
            results['model_metadata'] = {
                'base_model': cfg['model']['name'],
                'checkpoint_path': checkpoint_path if not (use_base_model or use_teacher_model) else None,
                'use_base_model': use_base_model,
                'use_teacher_model': use_teacher_model,
                'medical_only': medical_only,
                'model_identifier': model_id,
                'evaluation_timestamp': datetime.now().isoformat(),
                'size_reduced': cfg['model']['student'].get('reduce_size', False) if not use_teacher_model else False,
                'size_reduction_factor': cfg['model']['student'].get('size_reduction_factor', None) if cfg['model']['student'].get('reduce_size', False) and not use_teacher_model else None
            }
            
            # Save updated results with metadata
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to: {output_file}")
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
    parser.add_argument('--use_base_model', action='store_true', help='Use base HF model with student config instead of checkpoint')
    parser.add_argument('--use_teacher_model', action='store_true', help='Use teacher model (full-size, unquantized) for evaluation')
    parser.add_argument('--medical_only', action='store_true', help='Evaluate only medical MMLU tasks')
    args = parser.parse_args()
    
    # Validation: only one model type can be selected
    model_flags = [args.use_base_model, args.use_teacher_model, bool(args.checkpoint)]
    if sum(model_flags) != 1:
        parser.error("Exactly one of --checkpoint, --use_base_model, or --use_teacher_model must be provided")
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get model info and run evaluation
    model_path = load_model_info(args.checkpoint, cfg, args.use_base_model, args.use_teacher_model)
    results = run_mmlu_eval(model_path, args.output_dir, cfg, args.checkpoint, args.use_base_model, args.use_teacher_model, args.medical_only)
    
    if results:
        print("\nMMLU Evaluation Results:")
        print(f"Model: {results['model_metadata']['model_identifier']}")
        if 'results' in results:
            for task, metrics in results['results'].items():
                if 'acc' in metrics:
                    print(f"{task}: {metrics['acc']*100:.2f}%")
        else:
            print("Results format unexpected. Check the output file for details.")

if __name__ == "__main__":
    main()
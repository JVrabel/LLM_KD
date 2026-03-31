import argparse
import yaml
import os
from pathlib import Path
import subprocess
import json
import shutil
import tempfile
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def create_temp_model_dir(temp_dir=None):
    temp_root = Path(temp_dir) if temp_dir else Path(".")
    temp_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="temp_model_for_eval_", dir=str(temp_root)))

def load_model_info(checkpoint_path, cfg, use_base_model=False, use_teacher_model=False, temp_dir=None):
    """Prepare model information for evaluation - FIXED VERSION"""
    if use_teacher_model:
        temp_model_path = create_temp_model_dir(temp_dir)

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
        reduce_size = cfg['model']['student'].get('reduce_size', False)
        factor = cfg['model']['student'].get('size_reduction_factor', 1)

        # Avoid exporting and reloading the model when evaluating the unchanged base model.
        if not reduce_size or factor == 1:
            print("Using base model directly; skipping temporary export")
            return cfg['model']['name'], None

        temp_model_path = create_temp_model_dir(temp_dir)
        if reduce_size:
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
        temp_model_path = create_temp_model_dir(temp_dir)
        
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
    
    return str(temp_model_path.absolute()), str(temp_model_path.absolute())

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

def run_mmlu_eval(model_path, output_dir, cfg, checkpoint_path, use_base_model, use_teacher_model, medical_only=False, temp_model_dir=None):
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
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
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
        if temp_model_dir and Path(temp_model_dir).exists():
            shutil.rmtree(temp_model_dir)

def run_custom_eval(model, tokenizer, dataset_path):
    """Run evaluation on a custom dataset."""
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    results = []
    print(f"Evaluating on {len(dataset)} examples...")

    for i, item in enumerate(dataset):
        context = item.get("context", "")
        question = item["question"]
        choices = item["choices"]
        correct_answer = item["correct_answer"]

        # Format input for the model
        prompt = f"{context}\n{question}\n"
        for key, value in choices.items():
            prompt += f"{key}. {value}\n"
        prompt += "Answer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate one token
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.001,  # Greedy decoding
            )

        # Decode the generated token
        new_tokens = outputs[0][inputs.input_ids.shape[1] :]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Store the result
        is_correct = prediction.upper() == correct_answer.upper()
        results.append(
            {
                "question": question,
                "prediction": prediction,
                "correct_answer": correct_answer,
                "correct": is_correct,
            }
        )

        print(f"Example {i+1}:")
        print(f"Prompt:\n{prompt}")
        print(f"Prediction: {prediction}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Result: {'PASS' if is_correct else 'FAIL'}")
        print("-" * 30)

    # Calculate accuracy
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    print(f"\nTotal Accuracy: {accuracy:.2%}")

    return results

def main():
    parser = argparse.ArgumentParser(description='Run MMLU evaluation on a trained model')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation results')
    parser.add_argument('--use_base_model', action='store_true', help='Use base HF model with student config instead of checkpoint')
    parser.add_argument('--use_teacher_model', action='store_true', help='Use teacher model (full-size, unquantized) for evaluation')
    parser.add_argument('--medical_only', action='store_true', help='Evaluate only medical MMLU tasks')
    parser.add_argument('--temp_dir', type=str, default=None, help='Directory for temporary exported models')
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
    model_path, temp_model_dir = load_model_info(
        args.checkpoint,
        cfg,
        args.use_base_model,
        args.use_teacher_model,
        args.temp_dir,
    )
    results = run_mmlu_eval(
        model_path,
        args.output_dir,
        cfg,
        args.checkpoint,
        args.use_base_model,
        args.use_teacher_model,
        args.medical_only,
        temp_model_dir,
    )
    
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
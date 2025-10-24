import torch
import yaml
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pathlib import Path
import json
import os

def load_model_for_debug(checkpoint_path, cfg):
    """Load model the same way as eval.py but with debug output"""
    
    print("=== DEBUGGING MODEL LOADING ===")
    
    # Load base model config
    model_config = AutoConfig.from_pretrained(cfg['model']['name'])
    print(f"Base model config: {model_config}")
    
    # Check for size reduction
    if cfg['model']['student'].get('reduce_size', False):
        factor = cfg['model']['student']['size_reduction_factor']
        print(f"Reducing model size by factor of {factor}")
        print(f"Original hidden_size: {model_config.hidden_size}")
        print(f"Original intermediate_size: {model_config.intermediate_size}")
        print(f"Original num_attention_heads: {getattr(model_config, 'num_attention_heads', 'N/A')}")
        
        model_config.hidden_size = model_config.hidden_size // factor
        model_config.intermediate_size = model_config.intermediate_size // factor
        if hasattr(model_config, 'num_attention_heads'):
            model_config.num_attention_heads = model_config.num_attention_heads // factor
            
        print(f"New hidden_size: {model_config.hidden_size}")
        print(f"New intermediate_size: {model_config.intermediate_size}")
        print(f"New num_attention_heads: {getattr(model_config, 'num_attention_heads', 'N/A')}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg['model']['name'],
        config=model_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Debug checkpoint contents
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'student_model_state_dict' in checkpoint:
        state_dict = checkpoint['student_model_state_dict']
        print(f"Student model state dict keys: {len(state_dict.keys())}")
        print(f"First few keys: {list(state_dict.keys())[:10]}")
    else:
        print("Warning: 'student_model_state_dict' not found in checkpoint!")
        return None, None
    
    # Check model parameter names vs checkpoint keys
    model_params = set(name for name, _ in model.named_parameters())
    checkpoint_keys = set(state_dict.keys())
    
    print(f"Model parameters: {len(model_params)}")
    print(f"Checkpoint keys: {len(checkpoint_keys)}")
    
    missing_in_checkpoint = model_params - checkpoint_keys
    missing_in_model = checkpoint_keys - model_params
    
    if missing_in_checkpoint:
        print(f"Missing in checkpoint: {missing_in_checkpoint}")
    if missing_in_model:
        print(f"Missing in model: {missing_in_model}")
    
    # Handle weight tying for LLaMA
    if 'lm_head.weight' in state_dict:
        print("Found lm_head.weight in checkpoint")
        print("Handling weight tying...")
        state_dict['model.embed_tokens.weight'] = state_dict['lm_head.weight'].clone()
    
    # Load state dict
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Load result - Missing keys: {len(missing_keys)}")
        print(f"Load result - Unexpected keys: {len(unexpected_keys)}")
        if missing_keys:
            print(f"Missing keys: {missing_keys[:10]}...")  # Show first 10
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys[:10]}...")  # Show first 10
    except Exception as e:
        print(f"Error loading state dict: {e}")
        return None, None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def test_model_responses(model, tokenizer, test_prompts):
    """Test model responses on various prompts"""
    print("\n=== TESTING MODEL RESPONSES ===")
    
    model.eval()
    device = next(model.parameters()).device
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,  # Use greedy decoding for consistency
                pad_token_id=tokenizer.pad_token_id,
                temperature=1.0,
                top_p=1.0
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        print(f"Response: {response}")
        
        # Also test logits directly
        model_outputs = model(**inputs)
        logits = model_outputs.logits[0, -1, :]  # Last token logits
        top_k_indices = torch.topk(logits, k=5).indices
        top_k_tokens = [tokenizer.decode(idx) for idx in top_k_indices]
        top_k_probs = torch.softmax(logits, dim=-1)[top_k_indices]
        
        print(f"Top 5 next tokens: {list(zip(top_k_tokens, top_k_probs.tolist()))}")

def compare_with_base_model(cfg, test_prompts):
    """Compare with base model performance"""
    print("\n=== COMPARING WITH BASE MODEL ===")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg['model']['name'],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model.eval()
    device = next(base_model.parameters()).device
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Base Model Test {i+1} ---")
        print(f"Prompt: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        print(f"Base Model Response: {response}")

def test_mmlu_style_prompts(model, tokenizer):
    """Test with MMLU-style multiple choice prompts"""
    print("\n=== TESTING MMLU-STYLE PROMPTS ===")
    
    mmlu_prompts = [
        "The following are multiple choice questions (with answers) about anatomy.\n\nWhich of the following is the largest organ in the human body?\nA. Heart\nB. Liver\nC. Skin\nD. Brain\nAnswer:",
        "The following are multiple choice questions (with answers) about physics.\n\nWhat is the speed of light in vacuum?\nA. 3 × 10^8 m/s\nB. 3 × 10^6 m/s\nC. 3 × 10^10 m/s\nD. 3 × 10^4 m/s\nAnswer:",
        "The following are multiple choice questions (with answers) about mathematics.\n\nWhat is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:"
    ]
    
    model.eval()
    device = next(model.parameters()).device
    
    for prompt in mmlu_prompts:
        print(f"\nMMLU Prompt: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Get logits for the next token
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
        
        # Check probabilities for A, B, C, D tokens
        answer_tokens = ['A', 'B', 'C', 'D']
        answer_token_ids = [tokenizer.encode(token, add_special_tokens=False)[0] for token in answer_tokens]
        answer_logits = logits[answer_token_ids]
        answer_probs = torch.softmax(answer_logits, dim=-1)
        
        print("Answer probabilities:")
        for token, prob in zip(answer_tokens, answer_probs):
            print(f"  {token}: {prob:.4f}")
        
        predicted_answer = answer_tokens[torch.argmax(answer_probs)]
        print(f"Predicted answer: {predicted_answer}")
        
        # Also generate freely
        generation_outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        
        generated_text = tokenizer.decode(
            generation_outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        print(f"Generated continuation: '{generated_text}'")

def test_mmlu_likelihood_method(model, tokenizer):
    """Test using proper MMLU likelihood evaluation method"""
    print("\n=== TESTING MMLU LIKELIHOOD METHOD ===")
    
    test_questions = [
        {
            "prompt": "The following are multiple choice questions (with answers) about anatomy.\n\nWhich of the following is the largest organ in the human body?\nA. Heart\nB. Liver\nC. Skin\nD. Brain\nAnswer:",
            "correct": "C",
            "domain": "anatomy"
        },
        {
            "prompt": "The following are multiple choice questions (with answers) about medicine.\n\nWhat is the most common cause of acute pancreatitis?\nA. Gallstones\nB. Alcohol\nC. Trauma\nD. Medications\nAnswer:",
            "correct": "A",  # Gallstones are most common cause
            "domain": "medicine"
        },
        {
            "prompt": "The following are multiple choice questions (with answers) about medicine.\n\nWhich hormone regulates blood glucose levels?\nA. Cortisol\nB. Insulin\nC. Thyroid hormone\nD. Growth hormone\nAnswer:",
            "correct": "B",
            "domain": "medicine"
        }
    ]
    
    model.eval()
    device = next(model.parameters()).device
    
    correct_count = 0
    total_count = len(test_questions)
    
    for i, question in enumerate(test_questions):
        print(f"\n--- Question {i+1} ({question['domain']}) ---")
        print(question['prompt'])
        
        # Method 1: Likelihood evaluation (proper MMLU method)
        predicted_choice, choice_logprobs = mmlu_likelihood_evaluation(
            model, tokenizer, question['prompt']
        )
        
        print(f"Predicted: {predicted_choice}, Correct: {question['correct']}")
        print(f"Log probabilities: {dict(zip(['A', 'B', 'C', 'D'], choice_logprobs))}")
        
        if predicted_choice == question['correct']:
            correct_count += 1
            print("✓ CORRECT")
        else:
            print("✗ INCORRECT")
    
    accuracy = correct_count / total_count
    print(f"\nOverall Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    return accuracy

def mmlu_likelihood_evaluation(model, tokenizer, prompt, choices=['A', 'B', 'C', 'D']):
    """
    Proper MMLU-style likelihood evaluation
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Encode each choice and compute likelihood
    choice_logprobs = []
    
    for choice in choices:
        # Method: Compare likelihood of the choice token given the context
        context = prompt  # Everything up to "Answer:"
        choice_text = " " + choice  # The actual choice token
        
        # Tokenize context
        context_tokens = tokenizer(context, return_tensors="pt").to(device)
        
        # Tokenize context + choice
        full_tokens = tokenizer(context + choice_text, return_tensors="pt").to(device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = model(**full_tokens)
            logits = outputs.logits
        
        # Get the logit for the choice token
        choice_token_ids = tokenizer.encode(choice_text, add_special_tokens=False)
        
        if len(choice_token_ids) == 1:
            choice_token_id = choice_token_ids[0]
            # Position where choice token appears
            choice_position = len(context_tokens.input_ids[0]) - 1
            
            # Get log probability of this specific token
            log_probs = torch.log_softmax(logits[0, choice_position, :], dim=-1)
            choice_logprob = log_probs[choice_token_id].item()
        else:
            # Handle multi-token choices (shouldn't happen with A,B,C,D but just in case)
            choice_logprob = 0
            for j, token_id in enumerate(choice_token_ids):
                pos = len(context_tokens.input_ids[0]) + j - 1
                log_probs = torch.log_softmax(logits[0, pos, :], dim=-1)
                choice_logprob += log_probs[token_id].item()
        
        choice_logprobs.append(choice_logprob)
        print(f"Choice {choice}: logprob = {choice_logprob:.4f}")
    
    # Select choice with highest log probability
    best_choice_idx = torch.argmax(torch.tensor(choice_logprobs))
    predicted_choice = choices[best_choice_idx]
    
    return predicted_choice, choice_logprobs

def main():
    parser = argparse.ArgumentParser(description='Debug MMLU evaluation issues')
    parser.add_argument('--checkpoint', type=str, default='/home/LIBS/vrabel/projects/apollo/best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='src/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load model
    model, tokenizer = load_model_for_debug(args.checkpoint, cfg)
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "A description of the bile reflux and its symptoms are the following:",
        "What is 2+2?",
        "The largest planet in our solar system is"
    ]
    
    # Run tests
    test_model_responses(model, tokenizer, test_prompts)
    test_mmlu_style_prompts(model, tokenizer)
    compare_with_base_model(cfg, test_prompts[:2])  # Compare first 2 prompts only
    test_mmlu_likelihood_method(model, tokenizer)

if __name__ == "__main__":
    main() 
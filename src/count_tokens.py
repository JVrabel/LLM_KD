import json
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm

def count_tokens(dataset_path, model_name="meta-llama/Llama-3.2-1B"):
    """
    Count tokens in a JSONL dataset using a specific tokenizer.
    
    Args:
        dataset_path (str): Path to the JSONL dataset file
        model_name (str): Name of the model/tokenizer to use
    
    Returns:
        dict: Statistics about token counts
    """
    # Load tokenizer
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize counters
    total_tokens = 0
    total_examples = 0
    min_tokens = float('inf')
    max_tokens = 0
    token_counts = []
    
    # Process the JSONL file
    print(f"Processing dataset: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        # Count lines first for progress bar
        lines = f.readlines()
        total_examples = len(lines)
        
        # Process each line
        for line in tqdm(lines, desc="Counting tokens"):
            try:
                # Parse JSON
                data = json.loads(line)
                
                # Extract text based on common JSON formats
                # Adjust this based on your specific JSON structure
                if isinstance(data, dict):
                    if 'text' in data:
                        text = data['text']
                    elif 'content' in data:
                        text = data['content']
                    elif 'instruction' in data and 'output' in data:
                        # For instruction-tuning format
                        text = data['instruction'] + " " + data.get('output', '')
                    else:
                        # If no standard field found, convert the whole JSON to string
                        text = json.dumps(data)
                else:
                    text = str(data)
                
                # Count tokens
                tokens = tokenizer.encode(text)
                num_tokens = len(tokens)
                
                # Update statistics
                total_tokens += num_tokens
                min_tokens = min(min_tokens, num_tokens)
                max_tokens = max(max_tokens, num_tokens)
                token_counts.append(num_tokens)
                
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line as JSON, skipping")
                continue
    
    # Calculate statistics
    avg_tokens = total_tokens / total_examples if total_examples > 0 else 0
    
    # Print results
    print("\n===== Token Count Statistics =====")
    print(f"Model/Tokenizer: {model_name}")
    print(f"Total examples: {total_examples}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per example: {avg_tokens:.2f}")
    print(f"Min tokens in an example: {min_tokens}")
    print(f"Max tokens in an example: {max_tokens}")
    print("=================================")
    
    return {
        "total_examples": total_examples,
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count tokens in a JSONL dataset")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="Path to the JSONL dataset file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="meta-llama/Llama-3.2-1B",
        help="Model name for the tokenizer to use"
    )
    
    args = parser.parse_args()
    count_tokens(args.dataset, args.model)
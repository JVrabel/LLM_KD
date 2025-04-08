import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os

def generate_response(prompt, model_name="meta-llama/Llama-3.2-1B", checkpoint_path=None, max_new_tokens=150):
    """
    Loads a Hugging Face model, optionally loads weights from a checkpoint,
    and generates a response to the given prompt.

    Args:
        prompt (str): The input text prompt.
        model_name (str): The name/path of the base Hugging Face model architecture.
        checkpoint_path (str, optional): Path to the .pt checkpoint file. Defaults to None.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        str: The generated text response.
    """
    print(f"Loading base model architecture: {model_name}...")
    try:
        # Load tokenizer from the base model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token")

        # Determine device and dtype
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        print(f"Using device: {device}, dtype: {dtype}")

        # Load base model architecture
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        print("Base model loaded.")

        # Load weights from checkpoint if provided
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                print(f"Loading weights from checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                # Adjust key if necessary (e.g., if saved within a dict)
                state_dict_key = 'student_model_state_dict' # Common key in your training scripts
                if state_dict_key in checkpoint:
                    state_dict = checkpoint[state_dict_key]
                    print(f"Using state dict key: '{state_dict_key}'")
                else:
                     # Try loading the whole checkpoint as state_dict if key not found
                    state_dict = checkpoint
                    print("Checkpoint key 'student_model_state_dict' not found, attempting to load entire checkpoint as state_dict.")

                # Load the state dictionary
                load_result = model.load_state_dict(state_dict, strict=False) # Use strict=False initially for flexibility
                print(f"Checkpoint loading result: {load_result}")
                if load_result.missing_keys:
                    print(f"Warning: Missing keys: {load_result.missing_keys}")
                if load_result.unexpected_keys:
                    print(f"Warning: Unexpected keys: {load_result.unexpected_keys}")
            else:
                print(f"Warning: Checkpoint path not found: {checkpoint_path}. Using base model weights.")

        # Move model to device and set to eval mode
        model.to(device)
        model.eval()
        print("Model ready.")

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]

        # Generate response
        print("Generating response...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode and return response
        response = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
        print("Generation complete.")
        return response

    except Exception as e:
        print(f"An error occurred: {e}")
        # Print traceback for more details
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a Hugging Face model, optionally loading a checkpoint.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the concept of knowledge distillation in machine learning in simple terms:",
        help="The input prompt for the model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="The base Hugging Face model name/path to load the architecture from."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the .pt checkpoint file to load weights from (optional)."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=150,
        help="Maximum number of new tokens to generate."
    )
    args = parser.parse_args()

    generated_text = generate_response(args.prompt, args.model, args.checkpoint, args.max_tokens)

    if generated_text:
        print("\n" + "="*20 + " Prompt " + "="*20)
        print(args.prompt)
        print("\n" + "="*20 + " Response " + "="*20)
        print(generated_text)
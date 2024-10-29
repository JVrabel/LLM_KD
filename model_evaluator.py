import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
import os

class ModelEvaluator:
    def __init__(self, model_name, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self._load_model(model_name, checkpoint_path)

    def _load_model(self, model_name, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Load the configuration
        config = LlamaConfig.from_pretrained(model_name)
        config.num_hidden_layers = config.num_hidden_layers // 2
        config.num_attention_heads = config.num_attention_heads // 1
        config.hidden_size = config.hidden_size // 1
        config.intermediate_size = config.intermediate_size // 2

        # Create the model with the modified configuration
        model = LlamaForCausalLM(config)

        # Load the state dict
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['student_model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"Loaded model from checkpoint: {checkpoint_path}")
        return model

    def generate(self, prompt, max_new_tokens=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, top_k=50, top_p=0.95)
        
        # Decode only the newly generated tokens
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def get_next_token_probability(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        next_token_id = torch.argmax(probs).item()
        next_token = self.tokenizer.decode([next_token_id])
        next_token_prob = probs[next_token_id].item()
        
        return {
            "next_token": next_token,
            "probability": next_token_prob
        }

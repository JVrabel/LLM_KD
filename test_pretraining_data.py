import os
import requests
import json
import chardet
import random
from transformers import AutoTokenizer, LlamaForCausalLM
from model_evaluator import ModelEvaluator
import datetime
import torch
import numpy as np
from tqdm import tqdm

def download_book(url, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    book_id = url.split('/')[-1]
    txt_url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"

    response = requests.get(txt_url)
    if response.status_code == 200:
        filename = os.path.join(folder, f"{book_id}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Downloaded: {filename}")
        return filename
    else:
        print(f"Failed to download: {txt_url}")
        return None

def create_eval_set(input_file, output_file):
    try:
        # Detect the file encoding
        with open(input_file, 'rb') as raw_file:
            raw_data = raw_file.read()
            result = chardet.detect(raw_data)
            file_encoding = result['encoding']

        # Read the file with the detected encoding
        with open(input_file, "r", encoding=file_encoding) as file:
            file_contents = file.read()
        
        # Write to JSON file
        with open(output_file, "w", encoding='utf-8') as file:
            data = {"text": file_contents}
            json.dump(data, file, ensure_ascii=False)
        print(f"Successfully processed file: {input_file}")
        print(f"Evaluation data saved to: {output_file}")
    except Exception as e:
        print(f"Error processing file {input_file}: {e}")

def create_test_datasets(book_urls):
    input_folder = "input"
    output_folder = "data/test_datasets"
    os.makedirs(output_folder, exist_ok=True)
    
    test_datasets = []
    for i, url in enumerate(book_urls, 1):
        book_file = download_book(url, input_folder)
        if book_file:
            output_file = os.path.join(output_folder, f"test_dataset_{i}.jsonl")
            create_eval_set(book_file, output_file)
            test_datasets.append(output_file)
    
    return test_datasets

def extract_prompts(file_path, num_prompts=10, max_length=100):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text = data['text']
    words = text.split()
    prompts = []
    
    for _ in range(num_prompts):
        start = random.randint(0, max(0, len(words) - max_length - 50))
        prompt = ' '.join(words[start:start + max_length])
        continuation = ' '.join(words[start + max_length:start + max_length + 50])
        prompts.append((prompt, continuation))
    
    return prompts

def evaluate_models(model1, model2, teacher_model, test_datasets, num_prompts_per_dataset=10):
    results = {}
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    
    for dataset in test_datasets:
        dataset_results = []
        prompts = extract_prompts(dataset, num_prompts_per_dataset)
        
        for prompt, ground_truth in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model1.device)
            
            result = {
                "prompt": prompt,
                "ground_truth": ground_truth,
                "model1_completion": model1.generate(prompt, max_new_tokens=50),
                "model2_completion": model2.generate(prompt, max_new_tokens=50),
                "teacher_completion": tokenizer.decode(teacher_model.generate(**inputs, max_new_tokens=50)[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True),
            }
            dataset_results.append(result)
        
        results[dataset] = dataset_results
    
    return results

def compute_statistics(model, prompts, tokenizer):
    next_token_acc = []
    perplexities = []
    
    for prompt, ground_truth in tqdm(prompts):
        inputs = tokenizer(prompt + ground_truth, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits[:, :-1, :]
        labels = inputs.input_ids[:, 1:]
        
        # Next token prediction accuracy
        predictions = torch.argmax(logits, dim=-1)
        acc = (predictions == labels).float().mean().item()
        next_token_acc.append(acc)
        
        # Perplexity
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        perplexity = torch.exp(loss).item()
        perplexities.append(perplexity)
    
    return {
        "next_token_accuracy": np.mean(next_token_acc),
        "perplexity": np.mean(perplexities)
    }

def evaluate_models_comprehensive(model1, model2, teacher_model, test_datasets, num_prompts_per_dataset=100):
    results = {}
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    
    for dataset in test_datasets:
        prompts = extract_prompts(dataset, num_prompts_per_dataset)
        
        results[dataset] = {
            "model1": compute_statistics(model1.model, prompts, tokenizer),
            "model2": compute_statistics(model2.model, prompts, tokenizer),
            "teacher": compute_statistics(teacher_model, prompts, tokenizer)
        }
    
    return results

def save_results(results, comprehensive_results, output_dir, model1_path, model2_path, test_datasets):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"{timestamp}_models")
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "timestamp": timestamp,
        "model1_checkpoint": model1_path,
        "model2_checkpoint": model2_path,
        "test_datasets": test_datasets
    }

    with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    for dataset, dataset_results in results.items():
        dataset_name = os.path.basename(dataset).split('.')[0]
        with open(os.path.join(output_dir, f"{dataset_name}_results.json"), 'w', encoding='utf-8') as f:
            json.dump(dataset_results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "comprehensive_results.json"), 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation results and metadata saved to: {output_dir}")

def main():
    book_urls = [
        "https://www.gutenberg.org/ebooks/4391", # prinsiples of philosophy, Descartes
        "https://www.gutenberg.org/ebooks/27639", # new vegetarian dishes
        "https://www.gutenberg.org/ebooks/28553" # how it works (Technology)
    ]
    test_datasets = create_test_datasets(book_urls)
    
    model_name = "meta-llama/Llama-3.2-1B"
    checkpoint_path1 = "runs/kd_experiment/run_20241027_020053_cook/checkpoints/best_model.pt"
    checkpoint_path2 = "runs/kd_experiment/run_20241027_232646_phil/checkpoints/best_model.pt"

    evaluator1 = ModelEvaluator(model_name, checkpoint_path1)
    evaluator2 = ModelEvaluator(model_name, checkpoint_path2)
    
    # Load the teacher model
    teacher_model = LlamaForCausalLM.from_pretrained(model_name)
    teacher_model.to(evaluator1.device)
    teacher_model.eval()

    results = evaluate_models(evaluator1, evaluator2, teacher_model, test_datasets)
    comprehensive_results = evaluate_models_comprehensive(evaluator1, evaluator2, teacher_model, test_datasets)
    
    save_results(results, comprehensive_results, "results/eval", checkpoint_path1, checkpoint_path2, test_datasets)

if __name__ == "__main__":
    main()

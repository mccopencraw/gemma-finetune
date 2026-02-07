"""
Inference Script for Fine-tuned Gemma 3 Model
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
import CONFIG

def load_model(model_path=CONFIG.MODEL_SAVE_PATH):
    """Load the fine-tuned model"""
    print(f"Loading model from {model_path}...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=CONFIG.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer

def classify_article(model, tokenizer, article_text):
    """Classify a single article"""
    
    prompt = f"""### Instruction:
You are a news article classifier. Given a news article, classify it into one of these categories:
1. communicable disease outbreak
2. food poisoning
3. vaccine
4. medication research on communicable disease
5. public response to disease outbreak

### Article:
{article_text[:1500]}

### Response:
"""
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response part
    response = response.split("### Response:")[-1].strip()
    
    # Clean up
    for cat in CONFIG.CATEGORIES:
        if cat.lower() in response.lower():
            return cat
    
    # Try to extract first meaningful word
    return response.split()[0] if response else "unknown"

def classify_batch(model, tokenizer, articles, show_confidence=False):
    """Classify multiple articles"""
    results = []
    
    for i, article in enumerate(articles):
        category = classify_article(model, tokenizer, article)
        results.append({
            "index": i,
            "category": category,
            "text_preview": article[:100] + "..."
        })
        print(f"[{i+1}/{len(articles)}] {category}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Classify news articles")
    parser.add_argument("--text", type=str, help="Single article text to classify")
    parser.add_argument("--file", type=str, help="JSONL file with articles to classify")
    parser.add_argument("--model", type=str, default=CONFIG.MODEL_SAVE_PATH,
                        help="Path to fine-tuned model")
    parser.add_argument("--output", type=str, help="Output file for results")
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    if args.text:
        # Single article
        category = classify_article(model, tokenizer, args.text)
        print(f"\n{'='*50}")
        print(f"Article: {args.text[:100]}...")
        print(f"Category: {category}")
        print(f"{'='*50}")
        
    elif args.file:
        # Batch from file
        articles = []
        with open(args.file, 'r') as f:
            for line in f:
                data = json.loads(line)
                articles.append(data.get('text', data.get('content', '')))
        
        print(f"\nClassifying {len(articles)} articles...")
        results = classify_batch(model, tokenizer, articles)
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    else:
        print("Usage:")
        print("  python predict.py --text 'Your article text here'")
        print("  python predict.py --file articles.jsonl --output results.jsonl")

if __name__ == "__main__":
    main()

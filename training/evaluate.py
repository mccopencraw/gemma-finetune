"""
Evaluation Script for Fine-tuned Model
"""

import os
import json
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import CONFIG

def load_model(model_path=CONFIG.MODEL_SAVE_PATH):
    """Load the fine-tuned model"""
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
You are a news article classifier.

### Article:
{article_text[:1500]}

### Response:
"""
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:")[-1].strip().lower()
    
    # Match category
    for cat in CONFIG.CATEGORIES:
        if cat.lower() in response:
            return cat
    return "unknown"

def evaluate(test_path=CONFIG.TEST_DATA_PATH, model_path=CONFIG.MODEL_SAVE_PATH):
    """Evaluate the model on test data"""
    
    print("="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Load model
    model, tokenizer = load_model(model_path)
    
    # Load test data
    with open(test_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    print(f"\nEvaluating on {len(test_data)} test samples...")
    
    # Classify all test samples
    predictions = []
    ground_truth = []
    
    for item in test_data:
        pred = classify_article(model, tokenizer, item['text'])
        predictions.append(pred)
        ground_truth.append(item['category'])
    
    # Filter unknown
    valid_preds = [p for p in predictions if p != 'unknown']
    valid_truth = [gt for p, gt in zip(predictions, ground_truth) if p != 'unknown']
    
    print(f"\nValid predictions: {len(valid_preds)}/{len(predictions)}")
    
    # Classification report
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    report = classification_report(valid_truth, valid_preds, target_names=CONFIG.CATEGORIES)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(valid_truth, valid_preds, labels=CONFIG.CATEGORIES)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CONFIG.CATEGORIES,
                yticklabels=CONFIG.CATEGORIES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG.OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    print(f"\n✓ Confusion matrix saved to {CONFIG.OUTPUT_DIR}/confusion_matrix.png")
    
    # Save results
    results = {
        "test_samples": len(test_data),
        "valid_predictions": len(valid_preds),
        "report": report,
    }
    
    with open(os.path.join(CONFIG.OUTPUT_DIR, "evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return report

if __name__ == "__main__":
    evaluate()

"""
Google Colab Notebook - Gemma 3 Fine-tuning for Disease News Classification
==========================================================================

Quick Start:
1. Open: https://colab.research.google.com
2. New Notebook → Paste this code
3. Runtime → Change runtime type → GPU (T4 or V100)
4. Run cells one by one
"""

# ============================================================
# 📦 STEP 1: INSTALL DEPENDENCIES
# ============================================================
"""
# Mount Google Drive (optional - for saving checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# Install Unsloth and dependencies
!pip install -q unsloth transformers datasets trl peft accelerate bitsandbytes
!pip install -q torch torchvision torchaudio
!pip install -q xformers
!pip install -q sentencepiece protobuf

# For visualization (optional)
!pip install -q matplotlib seaborn scikit-learn
"""

# ============================================================
# 🔧 STEP 2: CLONE REPOSITORY
# ============================================================
"""
%cd /content
!git clone https://github.com/mccopencraw/gemma-finetune.git
%cd gemma-finetune

# Verify data exists
!ls -la data/processed/
"""

# ============================================================
# 🔑 STEP 3: SETUP API KEYS
# ============================================================
"""
import os

# Option 1: Set DeepSeek API key for data augmentation
# os.environ["DEEPSEEK_API_KEY"] = "your-api-key-here"

# Option 2: Use pre-labeled data (already collected)
# No API key needed for training

print("✓ Setup complete")
"""

# ============================================================
# 🤖 STEP 4: LOAD AND TRAIN MODEL
# ============================================================
"""
import torch
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
import json
from datetime import datetime

# Configuration
MODEL_NAME = "unsloth/gemma-3-1b-it"
OUTPUT_DIR = "./outputs"
MAX_SEQ_LENGTH = 512
LORA_RANK = 16
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
TRAIN_EPOCHS = 3

CATEGORIES = [
    "communicable disease outbreak",
    "food poisoning",
    "vaccine",
    "medication research on communicable disease",
    "public response to disease outbreak"
]

print("="*60)
print("Gemma 3 Fine-tuning for Disease News Classification")
print("="*60)
print(f"\n📊 Dataset: 1000 articles (200 per category)")
print(f"📦 Model: {MODEL_NAME}")
print(f"🔧 LoRA Rank: {LORA_RANK}")
print(f"🎯 Epochs: {TRAIN_EPOCHS}")

# Check GPU
print(f"\n🚀 Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

# Format prompts
def format_prompts(examples):
    texts = []
    for example in examples:
        text = f"""### Instruction:
You are a news article classifier. Classify this article into one of:
1. communicable disease outbreak
2. food poisoning
3. vaccine
4. medication research on communicable disease
5. public response to disease outbreak

### Article:
{example['text'][:1000]}

### Response:
{example['category']}"""
        texts.append(text)
    return {"text": texts}

# Load data
print("\n📂 Loading datasets...")
train_data = load_dataset("json", data_files="data/processed/train.jsonl", split="train")
val_data = load_dataset("json", data_files="data/processed/val.jsonl", split="train")

train_data = train_data.map(format_prompts, batched=True, remove_columns=list(train_data.features.keys()))
val_data = val_data.map(format_prompts, batched=True, remove_columns=list(val_data.features.keys()))

print(f"   Training samples: {len(train_data)}")
print(f"   Validation samples: {len(val_data)}")

# Load model
print("\n⬇️ Loading Gemma 3 model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# Add LoRA
print("🔧 Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing=True,
)

# Setup trainer
print("⚙️ Setting up trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=val_data,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=0.1,
        num_train_epochs=TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir=OUTPUT_DIR,
        save_steps=100,
        eval_steps=100,
        save_total_limit=3,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
    ),
)

# Train
print("\n" + "="*60)
print("🔥 STARTING TRAINING")
print("="*60)
start_time = datetime.now()

trainer_stats = trainer.train()

training_time = (datetime.now() - start_time).total_seconds()
print(f"\n✅ Training complete!")
print(f"   Time: {training_time/60:.1f} minutes")
print(f"   Final loss: {trainer_stats.metrics['train_loss']:.4f}")

# Save model
print("\n💾 Saving model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(f"{OUTPUT_DIR}/gemma-news-classifier")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/gemma-news-classifier")

# Save summary
summary = {
    "model": MODEL_NAME,
    "lora_rank": LORA_RANK,
    "train_samples": len(train_data),
    "val_samples": len(val_data),
    "epochs": TRAIN_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "final_loss": trainer_stats.metrics['train_loss'],
    "training_time_minutes": training_time / 60,
    "date": datetime.now().isoformat()
}
with open(f"{OUTPUT_DIR}/training_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n📁 Model saved to: {OUTPUT_DIR}/gemma-news-classifier")
print("="*60)
"""

# ============================================================
# 🧪 STEP 5: EVALUATE MODEL
# ============================================================
"""
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
import json

print("="*60)
print("MODEL EVALUATION")
print("="*60)

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./outputs/gemma-news-classifier",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Load test data
test_data = load_dataset("json", data_files="data/processed/test.jsonl", split="train")

# Categories
CATEGORIES = [
    "communicable disease outbreak",
    "food poisoning",
    "vaccine",
    "medication research on communicable disease",
    "public response to disease outbreak"
]

def classify(text):
    prompt = f"""### Instruction:
Classify this news article:

Article: {text[:800]}

Response:"""
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.1, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Response:")[-1].strip()

# Evaluate
predictions = []
ground_truth = []

for item in test_data:
    pred = classify(item['text'])
    predictions.append(pred)
    ground_truth.append(item['category'])

# Calculate accuracy
correct = sum(1 for p, g in zip(predictions, ground_truth) if p.lower() in g.lower() or g.lower() in p.lower())
accuracy = correct / len(predictions)

print(f"\n📊 Test Results:")
print(f"   Samples: {len(predictions)}")
print(f"   Accuracy: {accuracy*100:.1f}%")

# Classification report
from sklearn.metrics import classification_report
print("\n📋 Classification Report:")
print(classification_report(ground_truth, predictions, target_names=CATEGORIES))
"""

# ============================================================
# 🎯 STEP 6: INFERENCE DEMO
# ============================================================
"""
import torch
from unsloth import FastLanguageModel

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./outputs/gemma-news-classifier",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

CATEGORIES = [
    "communicable disease outbreak",
    "food poisoning",
    "vaccine",
    "medication research on communicable disease",
    "public response to disease outbreak"
]

def classify_news(article_text):
    prompt = f"""### Instruction:
Classify this news article into one category:
{', '.join(CATEGORIES)}

Article: {article_text}

Category:"""
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=30, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    result = response.split("Category:")[-1].strip()
    return result

# Test examples
test_articles = [
    "Hong Kong health officials confirm norovirus outbreak linked to raw oysters at local restaurants. Over 50 people have fallen ill with symptoms including vomiting and diarrhoea.",
    "Researchers at the University of Hong Kong develop new antiviral drug showing promise against multiple coronavirus variants in clinical trials.",
    "Hong Kong government announces expansion of flu vaccination program to all primary school students starting next month.",
    "Public panic buying reported at supermarkets following news of potential disease outbreak in neighbouring region.",
]

print("="*60)
print("INFERENCE DEMO")
print("="*60)

for i, article in enumerate(test_articles, 1):
    category = classify_news(article)
    print(f"\n[{i}] Article: {article[:60]}...")
    print(f"    → Category: {category}")
"""

print("\n" + "="*60)
print("✅ Colab Notebook Ready!")
print("="*60)
print("""
To run on Google Colab:
1. Go to https://colab.research.google.com
2. Create new notebook
3. Copy code from this file (colab_notebook.py)
4. Run cells sequentially

Estimated time on Colab T4:
- Training: 2-4 hours
- Evaluation: 5-10 minutes
""")

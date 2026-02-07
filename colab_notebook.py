"""
Google Colab Notebook - Gemma 3 Fine-tuning Pipeline
Copy and paste this into a Colab notebook or run as a script
"""

# ============================================================
# STEP 1: SETUP AND INSTALLATION
# ============================================================

"""
# Mount Google Drive (optional, for saving checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/yourusername/gemma-finetune.git
%cd gemma-finetune

# Install dependencies
!pip install -q -r requirements.txt
"""

# ============================================================
# STEP 2: CONFIGURATION
# ============================================================

import os
import json

# Set API keys
# os.environ["DEEPSEEK_API_KEY"] = "your-api-key-here"

# Configuration
MODEL_NAME = "unsloth/gemma-3-1b-it"  # Using 1B model (closest to 270m)
LORA_RANK = 16
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024
TRAIN_EPOCHS = 3

CATEGORIES = [
    "communicable disease outbreak",
    "food poisoning", 
    "vaccine",
    "medication research on communicable disease",
    "public response to disease outbreak"
]

print("✓ Configuration loaded")

# ============================================================
# STEP 3: DATA COLLECTION
# ============================================================

"""
# Run data collection
!python data_collection/collect_data.py
"""

# ============================================================
# STEP 4: DATA LABELING
# ============================================================

"""
# Run labeling with DeepSeek
!python labeling/label_data.py
"""

# ============================================================
# STEP 5: TRAINING
# ============================================================

"""
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from trl import SFTTrainer

print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing=True,
)

# Load data
train_data = load_dataset("json", data_files="data/processed/train.jsonl", split="train")
val_data = load_dataset("json", data_files="data/processed/val.jsonl", split="train")

def format_prompts(examples):
    texts = []
    for example in examples:
        text = f"""### Instruction:
You are a news article classifier.

### Article:
{example['text'][:1500]}

### Response:
{example['category']}"""
        texts.append(text)
    return {"text": texts}

train_data = train_data.map(format_prompts, batched=True, remove_columns=list(train_data.features.keys()))
val_data = val_data.map(format_prompts, batched=True, remove_columns=list(val_data.features.keys()))

print(f"Training samples: {len(train_data)}")

# Train
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
        output_dir="./outputs",
        save_steps=100,
        eval_steps=100,
        save_total_limit=3,
        evaluation_strategy="steps",
    ),
)

print("Starting training...")
trainer.train()

# Save
model.save_pretrained("./outputs/gemma-news-classifier")
tokenizer.save_pretrained("./outputs/gemma-news-classifier")
print("✓ Model saved!")
"""

# ============================================================
# STEP 6: INFERENCE
# ============================================================

"""
import torch
from unsloth import FastLanguageModel

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./outputs/gemma-news-classifier",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Test
test_article = """
Hong Kong health officials reported today a new outbreak of influenza A (H3N2) 
in a nursing home in Kowloon City district. At least 23 elderly residents have 
shown symptoms since Monday, and three have been hospitalized. The Centre for 
Health Protection is conducting investigations and has collected respiratory 
samples for testing. Vaccination status of affected residents is being verified.
"""

prompt = f"""### Instruction:
Classify this news article into one of these categories:
1. communicable disease outbreak
2. food poisoning
3. vaccine
4. medication research on communicable disease
5. public response to disease outbreak

### Article:
{test_article}

### Response:
"""

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=30, temperature=0.1, do_sample=False)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Classification: {response.split('### Response:')[-1].strip()}")
"""

print("\n" + "="*60)
print("Colab notebook template ready!")
print("="*60)

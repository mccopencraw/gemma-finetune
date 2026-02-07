"""
Gemma 3 Fine-tuning with Unsloth + LoRA
Optimized for Google Colab
"""

import os
import json
import torch
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
import CONFIG

# Setup
os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)

def format_prompts(examples):
    """Format examples for instruction tuning"""
    texts = []
    for example in examples:
        text = f"""### Instruction:
You are a news article classifier. Given a news article, classify it into one of these categories:
1. communicable disease outbreak
2. food poisoning
3. vaccine
4. medication research on communicable disease
5. public response to disease outbreak

### Article:
{example['text'][:1500]}

### Response:
{example['category']}"""
        texts.append(text)
    return {"text": texts}

def load_and_prepare_data():
    """Load and prepare datasets"""
    print("Loading datasets...")
    
    # Load train, val, test
    train_data = load_dataset("json", data_files=CONFIG.TRAIN_DATA_PATH, split="train")
    val_data = load_dataset("json", data_files=CONFIG.VAL_DATA_PATH, split="train")
    
    # Format
    train_data = train_data.map(format_prompts, batched=True, remove_columns=list(train_data.features.keys()))
    val_data = val_data.map(format_prompts, batched=True, remove_columns=list(val_data.features.keys()))
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    return train_data, val_data

def main():
    print("=" * 60)
    print("Gemma 3 Fine-tuning with Unsloth + LoRA")
    print("=" * 60)
    
    # Check device
    print(f"\nDevice: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model and tokenizer
    print("\n[1/4] Loading Gemma 3 with Unsloth...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG.MODEL_NAME,
        max_seq_length=CONFIG.MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect (bfloat16 if supported)
        load_in_4bit=True,  # Use 4-bit quantization for Colab
    )
    
    # Add LoRA adapters
    print("\n[2/4] Adding LoRA adapters...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG.LORA_RANK,
        lora_alpha=CONFIG.LORA_ALPHA,
        lora_dropout=CONFIG.LORA_DROPOUT,
        target_modules=CONFIG.LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing=True,
        use_rslora=True,
        loftq_config=None,
    )
    
    # Load data
    print("\n[3/4] Loading and preparing data...")
    train_data, val_data = load_and_prepare_data()
    
    # Training arguments
    print("\n[4/4] Setting up trainer...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        max_seq_length=CONFIG.MAX_SEQ_LENGTH,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=True,  # Pack multiple short sequences
        
        args=TrainingArguments(
            per_device_train_batch_size=CONFIG.BATCH_SIZE,
            per_device_eval_batch_size=CONFIG.BATCH_SIZE,
            gradient_accumulation_steps=CONFIG.GRADIENT_ACCUMULATION_STEPS,
            warmup_ratio=CONFIG.WARMUP_RATIO,
            num_train_epochs=CONFIG.TRAIN_EPOCHS,
            learning_rate=CONFIG.LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=CONFIG.LOGGING_STEPS,
            optim="adamw_8bit",
            weight_decay=CONFIG.WEIGHT_DECAY,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=CONFIG.OUTPUT_DIR,
            save_steps=CONFIG.SAVE_STEPS,
            eval_steps=CONFIG.EVAL_STEPS,
            save_total_limit=3,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",  # Disable wandb for Colab, use "wandb" if configured
            run_name=None,
        ),
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    trainer_stats = trainer.train()
    
    # Save model
    print("\nSaving model...")
    model.save_pretrained(CONFIG.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(CONFIG.MODEL_SAVE_PATH)
    
    # Training summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel saved to: {CONFIG.MODEL_SAVE_PATH}")
    print(f"Total training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"Final training loss: {trainer_stats.metrics['train_loss']:.4f}")
    
    # Save training config
    config_summary = {
        "model_name": CONFIG.MODEL_NAME,
        "lora_rank": CONFIG.LORA_RANK,
        "batch_size": CONFIG.BATCH_SIZE,
        "learning_rate": CONFIG.LEARNING_RATE,
        "epochs": CONFIG.TRAIN_EPOCHS,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "final_loss": trainer_stats.metrics['train_loss'],
        "training_time": trainer_stats.metrics['train_runtime'],
    }
    
    with open(os.path.join(CONFIG.OUTPUT_DIR, "training_summary.json"), 'w') as f:
        json.dump(config_summary, f, indent=2)

if __name__ == "__main__":
    main()

"""
Configuration file for Gemma 3 270m Fine-tuning Project
"""

# Model Configuration
# Using Gemma 3 1B (close to 270m scale) - works well on Colab free tier
MODEL_NAME = "unsloth/gemma-3-1b-it"

# LoRA Configuration
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Configuration (Optimized for Colab T4/V100)
BATCH_SIZE = 1  # Small batch for Colab GPU memory
GRADIENT_ACCUMULATION_STEPS = 8  # Accumulate gradients for effective larger batch
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512  # Shorter sequence for memory efficiency
TRAIN_EPOCHS = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
SAVE_STEPS = 100
EVAL_STEPS = 100

# Output Configuration
OUTPUT_DIR = "./outputs"
MODEL_SAVE_PATH = f"{OUTPUT_DIR}/gemma3-news-classifier"
LOG_DIR = f"{OUTPUT_DIR}/logs"

# DeepSeek API Configuration
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY_HERE"
DEEPSEEK_MODEL = "deepseek-chat"
LABELING_BATCH_SIZE = 10

# Classification Categories
CATEGORIES = [
    "communicable disease outbreak",
    "food poisoning",
    "vaccine",
    "medication research on communicable disease",
    "public response to disease outbreak"
]

CATEGORY_PROMPT = """Classify the following news article into ONE of these categories:
1. communicable disease outbreak
2. food poisoning
3. vaccine
4. medication research on communicable disease
5. public response to disease outbreak

Return ONLY the category name, nothing else."""

# Data Paths
RAW_DATA_DIR = "./data/raw"
PROCESSED_DATA_DIR = "./data/processed"
TRAIN_DATA_PATH = f"{PROCESSED_DATA_DIR}/train.jsonl"
VAL_DATA_PATH = f"{PROCESSED_DATA_DIR}/val.jsonl"
TEST_DATA_PATH = f"{PROCESSED_DATA_DIR}/test.jsonl"

# Training Data Split Ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

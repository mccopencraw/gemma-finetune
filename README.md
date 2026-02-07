# Gemma 3 270m Fine-tuning for News Classification
# Categories: communicable disease outbreak, food poisoning, vaccine, medication research, public response

## Setup Instructions for Google Colab

### 1. Open Google Colab
Go to https://colab.research.google.com and create a new notebook

### 2. Mount Google Drive (optional, for saving checkpoints)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Clone this repository
```python
!git clone https://github.com/mccopencraw/gemma-finetune.git
%cd gemma-finetune
```

### 4. Install dependencies
```bash
!pip install -r requirements.txt
```

### 5. Run the complete pipeline
```bash
# Step 1: Collect data
!python data_collection/collect_data.py

# Step 2: Label data with DeepSeek
!python labeling/label_data.py

# Step 3: Train with Unsloth + LoRA
!python training/train.py

# Step 4: Test inference
!python inference/predict.py --text "your test article here"
```

## Project Structure
```
gemma-finetune/
├── data_collection/
│   └── collect_data.py      # Collect news articles
├── labeling/
│   └── label_data.py        # Label using DeepSeek API
├── training/
│   ├── train.py            # Unsloth + LoRA training
│   └── data_prep.py        # Prepare data format
├── inference/
│   └── predict.py          # Inference script
├── requirements.txt
├── README.md
└── CONFIG.py               # Configuration file
```

## API Keys Required
- `DEEPSEEK_API_KEY` - For data labeling
- (Optional) `HF_TOKEN` - For accessing Gemma model

## Notes
- Training uses Unsloth for memory-efficient LoRA
- Expected runtime on Colab T4: ~2-4 hours
- Model will be saved to `outputs/` directory

# Gemma 3 Fine-tuning for Disease News Classification 🎯

Fine-tune Gemma 3 1B model to classify news articles into 5 categories:
- Communicable disease outbreak
- Food poisoning
- Vaccine
- Medication research on communicable disease
- Public response to disease outbreak

## 📊 Dataset (1000 articles, balanced)
- 200 articles per category
- Train: 800 | Val: 100 | Test: 100

## 🚀 Quick Start on Google Colab

### Option A: Use Colab Notebook (Recommended)
1. Open: **https://colab.research.google.com**
2. New Notebook → Copy code from `colab_train_notebook.py`
3. Runtime → Change runtime type → **GPU (T4 or V100)**
4. Run cells sequentially

### Option B: Run Commands
```bash
# Clone repo
git clone https://github.com/mccopencraw/gemma-finetune.git
cd gemma-finetune

# Install dependencies
pip install -q unsloth transformers datasets trl peft accelerate bitsandbytes

# Train
python training/train.py
```

## ⏱️ Estimated Training Time
- **Colab T4 (free)**: ~3-4 hours
- **Colab A100 (paid)**: ~1 hour

## 📁 Project Structure
```
gemma-finetune/
├── data/
│   └── processed/
│       ├── train.jsonl (800 samples)
│       ├── val.jsonl (100 samples)
│       └── test.jsonl (100 samples)
├── training/
│   ├── train.py         # Main training script
│   └── evaluate.py       # Evaluation script
├── inference/
│   └── predict.py        # Inference script
├── colab_train_notebook.py  # Ready-to-run Colab notebook
└── CONFIG.py             # Configuration
```

## 🎯 Model Specs
- **Base Model**: Gemma 3 1B (Unsloth optimized)
- **Fine-tuning**: LoRA (Rank 16)
- **Quantization**: 4-bit
- **Sequence Length**: 512

## 📦 Hardware Requirements (if running locally)
- GPU: NVIDIA with 8GB+ VRAM
- RAM: 16GB+
- Storage: 10GB+

## 🔧 Key Files
- `training/train.py` - Main training script with Unsloth + LoRA
- `colab_train_notebook.py` - Copy-paste Colab notebook
- `inference/predict.py` - Test your fine-tuned model

## 📖 Citation
If you use this model, please cite:
```bibtex
@misc{gemma-finetune-disease-news,
  title={Gemma Fine-tuning for Disease News Classification},
  author={OpenClaw},
  year={2026},
  url={https://github.com/mccopencraw/gemma-finetune}
}
```

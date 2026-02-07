"""
Data Labeling Script using DeepSeek API
Labels collected articles into 5 categories
"""

import os
import json
import time
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import CONFIG

# Setup directories
os.makedirs(CONFIG.PROCESSED_DATA_DIR, exist_ok=True)

def load_raw_articles():
    """Load raw articles from data collection"""
    input_file = os.path.join(CONFIG.RAW_DATA_DIR, "raw_articles.jsonl")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run collect_data.py first.")
        return []
    
    articles = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    
    return articles

def deepseek_classify(text, categories, api_key):
    """Use DeepSeek to classify an article"""
    
    prompt = f"""{CONFIG.CATEGORY_PROMPT}

Article text:
{text[:2000]}  # Limit text length

Category:"""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": CONFIG.DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a news classification assistant. Return ONLY the category name, nothing else."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 50,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            category = result['choices'][0]['message']['content'].strip().lower()
            
            # Normalize category to match one of the categories
            for cat in categories:
                if cat.lower() in category or category in cat.lower():
                    return cat
            
            return category if category in [c.lower() for c in categories] else "unknown"
        else:
            print(f"API Error: {response.status_code}")
            return "unknown"
            
    except Exception as e:
        print(f"Error: {e}")
        return "unknown"

def classify_articles(articles, api_key, batch_size=CONFIG.LABELING_BATCH_SIZE):
    """Classify articles with progress tracking"""
    
    categories = CONFIG.CATEGORIES
    labeled = []
    
    print(f"\nClassifying {len(articles)} articles...")
    
    for i, article in enumerate(tqdm(articles)):
        # Combine title and content for classification
        text = f"Title: {article.get('title', '')}\n\nContent: {article.get('content', article.get('summary', ''))}"
        
        if not text.strip():
            continue
            
        category = deepseek_classify(text, categories, api_key)
        
        labeled_article = {
            'text': text,
            'category': category,
            'source': article.get('source', 'unknown'),
            'url': article.get('url', ''),
            'date': article.get('date', '')
        }
        labeled.append(labeled_article)
        
        # Rate limiting
        time.sleep(0.5)
        
        # Save progress every batch
        if (i + 1) % batch_size == 0:
            progress_file = os.path.join(CONFIG.PROCESSED_DATA_DIR, "labeled_progress.jsonl")
            with open(progress_file, 'a', encoding='utf-8') as f:
                for la in labeled[-batch_size:]:
                    f.write(json.dumps(la, ensure_ascii=False) + '\n')
    
    return labeled

def split_data(labeled_articles, train_ratio=CONFIG.TRAIN_RATIO, 
               val_ratio=CONFIG.VAL_RATIO, test_ratio=CONFIG.TEST_RATIO):
    """Split data into train/val/test sets"""
    
    import random
    random.seed(42)
    
    # Shuffle
    shuffled = labeled_articles.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]
    
    return train_data, val_data, test_data

def save_jsonl(data, filepath):
    """Save data as JSONL"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    print("=" * 60)
    print("Data Labeling with DeepSeek")
    print("=" * 60)
    
    # Check API key
    api_key = CONFIG.DEEPSEEK_API_KEY
    if api_key == "YOUR_DEEPSEEK_API_KEY_HERE":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("\nError: DeepSeek API key not found!")
        print("Set it in CONFIG.py or as DEEPSEEK_API_KEY environment variable")
        return
    
    # Load raw articles
    print("\nLoading raw articles...")
    articles = load_raw_articles()
    print(f"Loaded {len(articles)} articles")
    
    if not articles:
        return
    
    # Classify
    labeled = classify_articles(articles, api_key)
    
    # Filter unknown categories
    valid_categories = [c.lower() for c in CONFIG.CATEGORIES]
    labeled = [a for a in labeled if a['category'].lower() in valid_categories]
    print(f"\nValid labeled articles: {len(labeled)}")
    
    # Show distribution
    from collections import Counter
    distribution = Counter([a['category'] for a in labeled])
    print("\nCategory distribution:")
    for cat, count in sorted(distribution.items()):
        print(f"  {cat}: {count}")
    
    # Split data
    print("\nSplitting data...")
    train_data, val_data, test_data = split_data(labeled)
    
    # Save
    save_jsonl(train_data, CONFIG.TRAIN_DATA_PATH)
    save_jsonl(val_data, CONFIG.VAL_DATA_PATH)
    save_jsonl(test_data, CONFIG.TEST_DATA_PATH)
    
    print(f"\n✓ Training set: {len(train_data)} articles")
    print(f"✓ Validation set: {len(val_data)} articles")
    print(f"✓ Test set: {len(test_data)} articles")
    print(f"\nData saved to {CONFIG.PROCESSED_DATA_DIR}")
    print("\nLabeling complete!")

if __name__ == "__main__":
    main()

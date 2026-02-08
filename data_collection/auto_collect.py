"""
Automated News Collection and Labeling Pipeline
Collects and labels 1000+ news articles for disease classification
"""

import os
import json
import time
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import CONFIG

# API Key Check
DEEPSEEK_API_KEY = CONFIG.DEEPSEEK_API_KEY
if DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY_HERE":
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    print("ERROR: DeepSeek API key not found!")
    print("Set DEEPSEEK_API_KEY in CONFIG.py or as environment variable")
    exit(1)

CATEGORIES = CONFIG.CATEGORIES

SEARCH_QUERIES = [
    # Communicable disease outbreak
    "disease outbreak Hong Kong 2026",
    "influenza outbreak Asia 2026",
    "norovirus outbreak 2026",
    "measles outbreak 2026",
    "dengue fever outbreak 2026",
    "tuberculosis outbreak 2026",
    "respiratory virus outbreak 2026",
    "viral infection cluster 2026",
    "传染病爆发 香港 2026",
    "病毒感染 群聚 2026",
    
    # Food poisoning
    "food poisoning Hong Kong 2026",
    "食物中毒 香港 2026",
    "salmonella outbreak 2026",
    "norovirus food poisoning 2026",
    "食物安全 2026",
    "restaurant food poisoning 2026",
    "canned food contamination 2026",
    
    # Vaccine
    "vaccine Hong Kong 2026",
    "新冠疫苗 香港 2026",
    "flu vaccine 2026",
    "vaccination program 2026",
    "COVID vaccine booster 2026",
    "immunization Hong Kong 2026",
    "疫苗接种 2026",
    "mRNA vaccine 2026",
    "childhood vaccine 2026",
    
    # Medication research
    "antiviral drug research 2026",
    "COVID treatment 2026",
    "antibiotic development 2026",
    "new drug trial 2026",
    "clinical trial Hong Kong 2026",
    "药物研发 2026",
    "抗病毒药物 2026",
    "传染病药物 2026",
    "pharmaceutical research 2026",
    
    # Public response
    "public reaction disease outbreak 2026",
    "pandemic preparedness 2026",
    "health measures public 2026",
    "public fear disease 2026",
    "vaccine hesitancy 2026",
    "mask wearing public 2026",
    "public opinion COVID 2026",
    "防疫措施 公众 2026",
    "市民应对 疫情 2026",
]

# Source URLs to collect from
RSS_FEEDS = [
    "https://www.scmp.com/rss/2/feed/30372069/News.xml",
    "https://www.hkfp.com/rss",
    "https://rthk.hk/rss/xml/news/flh_en.xml",
    "https://www.thestandard.com.hk/rss/news",
]

NEWS_SITES = [
    {
        "name": "HK Centre for Health Protection",
        "base_url": "https://www.chp.gov.hk/en/features",
        "patterns": ["/news/", "/features/"]
    },
    {
        "name": "SCMP Health",
        "base_url": "https://www.scmp.com/news/hong-kong/health-environment",
        "patterns": ["/health-environment/"]
    },
    {
        "name": "HKFP Health",
        "base_url": "https://hongkongfp.com/category/hong-kong/health",
        "patterns": ["/category/hong-kong/health"]
    },
]

def save_articles(articles, filename):
    """Save articles to JSONL file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'a', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

def get_existing_count():
    """Get count of existing articles"""
    raw_file = os.path.join(CONFIG.RAW_DATA_DIR, "raw_articles.jsonl")
    if not os.path.exists(raw_file):
        return 0
    with open(raw_file, 'r') as f:
        return sum(1 for _ in f)

def main():
    print("="*60)
    print("Automated News Collection & Labeling Pipeline")
    print("="*60)
    print(f"\nTarget: 1000 articles")
    print(f"Current count: {get_existing_count()}")
    print("\nNote: This script uses web search to find articles.")
    print("For bulk collection, run in a loop or use Colab with the notebook.")
    print("\n" + "="*60)
    
    print("\nTo collect 1000 articles, use one of these methods:")
    print("\n1. COLAB - Run the notebook:")
    print("   !git clone https://github.com/mccopencraw/gemma-finetune.git")
    print("   cd gemma-finetune")
    print("   pip install -r requirements.txt")
    print("   # Then run cells in colab_notebook.py")
    
    print("\n2. LOCAL - Run collection script:")
    print("   python3 data_collection/collect_data.py")
    print("   python3 labeling/label_data.py")
    
    print("\n3. MANUAL - Add articles to data/raw/raw_articles.jsonl")
    print("   Then run labeling.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()

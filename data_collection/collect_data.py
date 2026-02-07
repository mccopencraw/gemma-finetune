"""
News Article Data Collection Script
Collects articles from: HK news, Outbreak News Today, CIDRAP
"""

import os
import json
import time
import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import CONFIG

# Setup directories
os.makedirs(CONFIG.RAW_DATA_DIR, exist_ok=True)

def collect_hk_news():
    """Collect from Hong Kong news sources"""
    articles = []
    
    # Hong Kong Centre for Health Protection
    urls = [
        "https://www.chp.gov.hk/en/features/102147.html",  # COVID-19
        "https://www.chp.gov.hk/en/features/66419.html",    # Communicable diseases
    ]
    
    for url in urls:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extract article links and summaries
            for link in soup.find_all('a', href=True):
                if '/en/features/' in link['href'] or '/en/news/' in link['href']:
                    articles.append({
                        'source': 'hk_chp',
                        'url': f"https://www.chp.gov.hk{link['href']}" if link['href'].startswith('/') else link['href'],
                        'title': link.get_text(strip=True)[:200],
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'collected_at': datetime.now().isoformat()
                    })
            time.sleep(1)
        except Exception as e:
            print(f"Error collecting HK news: {e}")
    
    print(f"Collected {len(articles)} HK articles")
    return articles

def collect_outbreak_news_today():
    """Collect from Outbreak News Today"""
    articles = []
    
    # RSS Feed
    rss_url = "https://outbreaknewstoday.com/feed/"
    
    try:
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries[:100]:  # Last 100 articles
            articles.append({
                'source': 'outbreak_news_today',
                'url': entry.get('link', ''),
                'title': entry.get('title', '')[:200],
                'summary': entry.get('summary', '')[:500],
                'date': entry.get('published', ''),
                'collected_at': datetime.now().isoformat()
            })
    except Exception as e:
        print(f"Error collecting Outbreak News Today: {e}")
    
    print(f"Collected {len(articles)} Outbreak News Today articles")
    return articles

def collect_cidrap():
    """Collect from CIDRAP (Center for Infectious Disease Research and Policy)"""
    articles = []
    
    # CIDRAP News RSS
    rss_urls = [
        "https://www.cidrap.umn.edu/news-rss",
        "https://www.cidrap.umn.edu/covid-19/news-rss",
        "https://www.cidrap.umn.edu/avian-influenza-bird-flu/news-rss",
    ]
    
    for rss_url in rss_urls:
        try:
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:50]:  # 50 articles per category
                articles.append({
                    'source': 'cidrap',
                    'url': entry.get('link', ''),
                    'title': entry.get('title', '')[:200],
                    'summary': entry.get('summary', '')[:500],
                    'date': entry.get('published', ''),
                    'collected_at': datetime.now().isoformat()
                })
            time.sleep(1)
        except Exception as e:
            print(f"Error collecting CIDRAP: {e}")
    
    print(f"Collected {len(articles)} CIDRAP articles")
    return articles

def fetch_article_content(article):
    """Fetch full content of an article"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(article['url'], headers=headers, timeout=30)
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Remove scripts and styles
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get main content
        content = soup.get_text(separator=' ', strip=True)
        content = ' '.join(content.split())[:3000]  # Limit to 3000 chars
        
        return {
            **article,
            'content': content
        }
    except Exception as e:
        print(f"Error fetching {article['url']}: {e}")
        return {
            **article,
            'content': article.get('summary', article.get('title', ''))
        }

def main():
    print("=" * 60)
    print("Starting Data Collection")
    print("=" * 60)
    
    all_articles = []
    
    # Collect from all sources
    print("\n[1/3] Collecting Hong Kong news...")
    all_articles.extend(collect_hk_news())
    
    print("\n[2/3] Collecting Outbreak News Today...")
    all_articles.extend(collect_outbreak_news_today())
    
    print("\n[3/3] Collecting CIDRAP articles...")
    all_articles.extend(collect_cidrap())
    
    # Fetch full content for each article
    print(f"\nFetching full content for {len(all_articles)} articles...")
    articles_with_content = []
    
    for i, article in enumerate(all_articles):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(all_articles)}")
        articles_with_content.append(fetch_article_content(article))
        time.sleep(0.5)  # Rate limiting
    
    # Save raw data
    output_file = os.path.join(CONFIG.RAW_DATA_DIR, "raw_articles.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in articles_with_content:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Saved {len(articles_with_content)} articles to {output_file}")
    print(f"\nData collection complete!")

if __name__ == "__main__":
    main()

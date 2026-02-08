"""
News Article Data Collection Script
Collects articles from reliable disease outbreak news sources
"""

import os
import json
import time
import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import CONFIG

# Setup directories
os.makedirs(CONFIG.RAW_DATA_DIR, exist_ok=True)

def collect_hk_news():
    """Collect from Hong Kong news sources"""
    articles = []
    
    # HK Centre for Health Protection
    sources = [
        {
            'source': 'hk_chp_covid',
            'url': 'https://www.chp.gov.hk/en/features/102147.html',
            'name': 'HK CHP - COVID-19'
        },
        {
            'source': 'hk_chp_outbreak',
            'url': 'https://www.chp.gov.hk/en/features/66419.html',
            'name': 'HK CHP - Communicable Diseases'
        },
        {
            'source': 'hk_news_health',
            'url': 'https://news.gov.hk/eng/categories/health.html',
            'name': 'HK Gov News - Health'
        }
    ]
    
    for src in sources:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(src['url'], headers=headers, timeout=30)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'lxml')
                
                # Extract article links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if ('/en/news/' in href or '/en/features/' in href) and 'chp.gov.hk' in src['url']:
                        full_url = f"https://www.chp.gov.hk{href}" if href.startswith('/') else href
                        articles.append({
                            'source': src['source'],
                            'url': full_url,
                            'title': link.get_text(strip=True)[:200],
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'collected_at': datetime.now().isoformat()
                        })
            
            time.sleep(1)
        except Exception as e:
            print(f"Error collecting {src['name']}: {e}")
    
    # Remove duplicates
    seen = set()
    unique_articles = []
    for a in articles:
        if a['url'] not in seen:
            seen.add(a['url'])
            unique_articles.append(a)
    
    print(f"Collected {len(unique_articles)} HK articles")
    return unique_articles

def collect_outbreak_news_today():
    """Collect from Outbreak News Today"""
    articles = []
    
    rss_feeds = [
        'https://outbreaknewstoday.com/feed/',
        'https://outbreaknewstoday.com/category/coronavirus/feed/',
        'https://outbreaknewstoday.com/category/avian-flu/feed/',
    ]
    
    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:30]:
                articles.append({
                    'source': 'outbreak_news_today',
                    'url': entry.get('link', ''),
                    'title': entry.get('title', '')[:200],
                    'summary': entry.get('summary', '')[:500],
                    'date': entry.get('published', '')[:10] if entry.get('published') else datetime.now().strftime('%Y-%m-%d'),
                    'collected_at': datetime.now().isoformat()
                })
            
            time.sleep(1)
        except Exception as e:
            print(f"Error collecting Outbreak News Today: {e}")
    
    print(f"Collected {len(articles)} Outbreak News Today articles")
    return articles

def collect_cidrap():
    """Collect from CIDRAP"""
    articles = []
    
    rss_feeds = [
        'https://www.cidrap.umn.edu/news-rss',
        'https://www.cidrap.umn.edu/covid-19/news-rss',
        'https://www.cidrap.umn.edu/avian-influenza-bird-flu/news-rss',
    ]
    
    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:30]:
                articles.append({
                    'source': 'cidrap',
                    'url': entry.get('link', ''),
                    'title': entry.get('title', '')[:200],
                    'summary': entry.get('summary', '')[:500],
                    'date': entry.get('published', '')[:10] if entry.get('published') else datetime.now().strftime('%Y-%m-%d'),
                    'collected_at': datetime.now().isoformat()
                })
            
            time.sleep(1)
        except Exception as e:
            print(f"Error collecting CIDRAP: {e}")
    
    print(f"Collected {len(articles)} CIDRAP articles")
    return articles

def collect_who():
    """Collect from WHO"""
    articles = []
    
    rss_feeds = [
        'https://www.who.int/rss-feeds/news-stand/en',
    ]
    
    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:30]:
                # Check if related to diseases
                title_lower = entry.get('title', '').lower()
                summary_lower = entry.get('summary', '').lower()
                keywords = ['disease', 'outbreak', 'flu', 'influenza', 'covid', 'vaccine', 'infection', 'virus', 'health']
                
                if any(kw in title_lower or kw in summary_lower for kw in keywords):
                    articles.append({
                        'source': 'who',
                        'url': entry.get('link', ''),
                        'title': entry.get('title', '')[:200],
                        'summary': entry.get('summary', '')[:500],
                        'date': entry.get('published', '')[:10] if entry.get('published') else datetime.now().strftime('%Y-%m-%d'),
                        'collected_at': datetime.now().isoformat()
                    })
            
            time.sleep(1)
        except Exception as e:
            print(f"Error collecting WHO: {e}")
    
    print(f"Collected {len(articles)} WHO articles")
    return articles

def fetch_article_content(article):
    """Fetch full content of an article"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(article['url'], headers=headers, timeout=30)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Remove scripts and styles
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            
            # Get main content
            content = soup.get_text(separator=' ', strip=True)
            content = ' '.join(content.split())[:3000]
            
            return {**article, 'content': content}
        
        return {**article, 'content': article.get('summary', article.get('title', ''))}
        
    except Exception as e:
        print(f"Error fetching {article['url']}: {e}")
        return {**article, 'content': article.get('summary', article.get('title', ''))}

def main():
    print("=" * 60)
    print("Starting Data Collection")
    print("=" * 60)
    
    all_articles = []
    
    print("\n[1/4] Collecting Hong Kong news...")
    all_articles.extend(collect_hk_news())
    
    print("\n[2/4] Collecting Outbreak News Today...")
    all_articles.extend(collect_outbreak_news_today())
    
    print("\n[3/4] Collecting CIDRAP articles...")
    all_articles.extend(collect_cidrap())
    
    print("\n[4/4] Collecting WHO articles...")
    all_articles.extend(collect_who())
    
    # Fetch full content
    print(f"\nFetching full content for {len(all_articles)} articles...")
    articles_with_content = []
    
    for i, article in enumerate(all_articles):
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(all_articles)}")
        articles_with_content.append(fetch_article_content(article))
        time.sleep(0.5)
    
    # Save raw data
    output_file = os.path.join(CONFIG.RAW_DATA_DIR, "raw_articles.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in articles_with_content:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Saved {len(articles_with_content)} articles to {output_file}")
    print(f"\nData collection complete!")

if __name__ == "__main__":
    main()

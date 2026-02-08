"""
Generate synthetic labeled articles using DeepSeek API
Quickly expands dataset to 1000+ samples through paraphrasing
"""

import os
import json
import time
import requests
import random
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import CONFIG

DEEPSEEK_API_KEY = CONFIG.DEEPSEEK_API_KEY
if DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY_HERE":
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

CATEGORIES = CONFIG.CATEGORIES

# Seed articles for augmentation
SEED_ARTICLES = {
    "communicable disease outbreak": [
        {
            "text": "Hong Kong health authorities have confirmed an outbreak of influenza A (H3N2) affecting multiple residential care homes in the New Territories. At least 45 elderly residents have developed symptoms including fever, cough, and respiratory distress since last Monday. The Centre for Health Protection has dispatched teams to investigate the source of infection and implement control measures. Vaccination status of affected individuals is being verified.",
            "category": "communicable disease outbreak"
        },
        {
            "text": "A cluster of norovirus cases has been reported among students at a secondary school in Kowloon, with over 30 individuals experiencing vomiting, diarrhoea, and abdominal pain. Health officials believe the outbreak may have originated from contaminated food at the school canteen. The affected students have been advised to stay home and seek medical attention if symptoms worsen.",
            "category": "communicable disease outbreak"
        },
        {
            "text": "The Centre for Health Protection is investigating a possible measles exposure incident at a shopping mall in Causeway Bay. An infected individual visited multiple locations while infectious, potentially exposing dozens of shoppers. Health authorities are contacting identified close contacts and offering post-exposure prophylaxis vaccination.",
            "category": "communicable disease outbreak"
        },
        {
            "text": "Dengue fever cases have increased significantly in the Kwai Shing estate area, with 12 confirmed cases reported this month. The Food and Environmental Hygiene Department has intensified mosquito control measures and conducted inspections to identify and eliminate breeding sites. Residents are advised to remove standing water and use insect repellent.",
            "category": "communicable disease outbreak"
        },
        {
            "text": "An outbreak of hand, foot, and mouth disease has affected multiple kindergartens in the Eastern District, with over 50 children showing symptoms including fever, mouth sores, and rashes. The Education Bureau has issued guidelines to affected schools to implement hygiene measures and temporarily suspend group activities.",
            "category": "communicable disease outbreak"
        },
    ],
    "food poisoning": [
        {
            "text": "Six customers have been hospitalized after dining at a seafood restaurant in Sai Ying Pun, presenting symptoms consistent with ciguatera fish poisoning. The Centre for Food Safety has collected food samples from the restaurant for laboratory analysis. Diners are advised to avoid consuming large reef fish from the South China Sea.",
            "category": "food poisoning"
        },
        {
            "text": "The Centre for Health Protection is investigating three separate food poisoning incidents linked to a popular sushi chain across Hong Kong. Over 40 customers reported symptoms of salmonella infection after consuming raw fish dishes. The affected restaurants have been instructed to suspend operations and undergo deep cleaning.",
            "category": "food poisoning"
        },
        {
            "text": "A food poisoning outbreak has been traced to contaminated pre-packaged sandwiches sold at convenience stores across the city. The Centre for Food Safety has issued a recall notice for the affected products, which were found to contain elevated levels of Staphylococcus aureus. Consumers who purchased the items are advised to discard them immediately.",
            "category": "food poisoning"
        },
        {
            "text": "More than 80 guests at a wedding banquet in Tsim Sha Tsui have fallen ill with symptoms of food poisoning, including severe vomiting and diarrhoea. Health officials suspect the source may be undercooked poultry served at the event. The catering company has cooperated with investigations and offered medical compensation to affected guests.",
            "category": "food poisoning"
        },
        {
            "text": "The Centre for Food Safety has issued a warning about potential histamine fish poisoning from consumption of improperly stored tuna at local supermarkets. Several consumers reported allergic-like reactions after eating canned tuna products. The affected batches have been removed from shelves pending further testing.",
            "category": "food poisoning"
        },
    ],
    "vaccine": [
        {
            "text": "The Hong Kong government has announced the expansion of the free seasonal influenza vaccination program to include all primary school students. The initiative aims to increase vaccination coverage and reduce flu-related absences and complications among children. Parents are encouraged to consent to vaccination through the school-based program.",
            "category": "vaccine"
        },
        {
            "text": "Clinical trials for a new universal influenza vaccine developed by researchers at the University of Hong Kong have shown promising results in phase 2 testing. The vaccine targets conserved regions of the virus and could provide long-lasting protection against multiple strains. Large-scale trials are planned for next year.",
            "category": "vaccine"
        },
        {
            "text": "Hong Kong has secured an additional 500,000 doses of the updated COVID-19 booster vaccine targeting the latest variants. The vaccination program will prioritize elderly residents, healthcare workers, and individuals with chronic conditions. Appointments can be booked through the government vaccination booking system.",
            "category": "vaccine"
        },
        {
            "text": "The Scientific Committee on Vaccine Preventable Diseases has recommended the inclusion of the human papillomavirus (HPV) vaccine in the routine childhood immunization schedule for both boys and girls. The government is considering the recommendation and conducting cost-effectiveness analysis before implementation.",
            "category": "vaccine"
        },
        {
            "text": "A new study has confirmed the high effectiveness of the combined COVID-19 and influenza vaccine developed by a local pharmaceutical company. Clinical results show robust immune response against both viruses with a favorable safety profile. Regulatory approval is expected within the next six months.",
            "category": "vaccine"
        },
    ],
    "medication research on communicable disease": [
        {
            "text": "Researchers at the Chinese University of Hong Kong have developed a new antiviral drug that shows effectiveness against multiple coronavirus variants. The drug, named CUHK-CoronaRx, inhibits viral replication by targeting a novel protein mechanism. Preclinical studies show promising results, with clinical trials planned to begin next year.",
            "category": "medication research on communicable disease"
        },
        {
            "text": "A breakthrough antibiotic developed by Hong Kong scientists has received fast-track approval for the treatment of multidrug-resistant tuberculosis. The new drug, named HK-TB-01, shows high efficacy against strains that are resistant to current treatments. The medication will be made available through the Hospital Authority.",
            "category": "medication research on communicable disease"
        },
        {
            "text": "Scientists have identified a promising monoclonal antibody therapy for the treatment of severe dengue fever. The antibody, discovered through a collaboration between the University of Hong Kong and international researchers, has shown ability to neutralize all four dengue virus serotypes in laboratory studies.",
            "category": "medication research on communicable disease"
        },
        {
            "text": "A new formulation of remdesivir with improved bioavailability has been developed by researchers in Hong Kong. The new delivery method allows for oral administration instead of intravenous infusion, making treatment more accessible. Clinical trials are underway to confirm efficacy in COVID-19 patients.",
            "category": "medication research on communicable disease"
        },
        {
            "text": "Hong Kong researchers have completed a successful phase 3 trial of a new treatment for chronic hepatitis B. The drug, which works by suppressing viral replication and stimulating immune response, achieved functional cure in a significant proportion of participants. Registration with regulatory authorities is in progress.",
            "category": "medication research on communicable disease"
        },
    ],
    "public response to disease outbreak": [
        {
            "text": "Following the announcement of a new influenza strain detected in the region, Hong Kong residents have rushed to obtain flu vaccinations at community clinics. Long queues have been reported at vaccination centres across the city. Pharmacies have also reported increased sales of face masks and antiviral medications.",
            "category": "public response to disease outbreak"
        },
        {
            "text": "A survey conducted by the University of Hong Kong has found that public confidence in the government's disease preparedness has increased significantly following recent health infrastructure investments. However, concerns remain about communication transparency during health emergencies. The findings will inform future public health communication strategies.",
            "category": "public response to disease outbreak"
        },
        {
            "text": "Social media platforms have seen a surge in discussions about disease preparedness following media reports of outbreaks in neighbouring regions. Health authorities have launched an online information campaign to counter misinformation and provide accurate guidance. Citizens are urged to rely on official sources for health information.",
            "category": "public response to disease outbreak"
        },
        {
            "text": "Hong Kong residents have begun stocking up on essential supplies as concerns about potential disease outbreaks grow. Retailers report increased sales of disinfectants, hand sanitizers, and non-perishable food items. The government has assured the public that supply chains remain stable and urged against panic buying.",
            "category": "public response to disease outbreak"
        },
        {
            "text": "Parents have expressed mixed reactions to new school guidelines for disease prevention implemented by the Education Bureau. While some welcome the measures as necessary precautions, others worry about disruption to their children's education. Schools have been provided with detailed protocols for managing suspected infection cases.",
            "category": "public response to disease outbreak"
        },
    ]
}

def paraphrase_with_deepseek(text, api_key):
    """Use DeepSeek to paraphrase an article"""
    prompt = f"""Paraphrase the following news article about a disease-related topic. 
Keep the same meaning, category, and factual information but change the wording and structure.

Original article:
{text}

Paraphrased article (keep the same category label):"""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": CONFIG.DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system", 
                "content": "You are a news article paraphrasing assistant. Return ONLY the paraphrased article text, nothing else."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "temperature": 0.8
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
            return result['choices'][0]['message']['content'].strip()
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def augment_dataset(target_count=1000):
    """Generate synthetic articles to reach target count"""
    
    if not DEEPSEEK_API_KEY:
        print("ERROR: DeepSeek API key not configured")
        return []
    
    all_articles = []
    current_count = 0
    augmentation_rounds = 0
    max_rounds = 100
    
    print("="*60)
    print("Dataset Augmentation for 1000 Articles")
    print("="*60)
    print(f"Target: {target_count} articles")
    
    # Load existing articles
    for split in ['train', 'val', 'test']:
        filepath = f"data/processed/{split}.jsonl"
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        all_articles.append(data)
                        current_count += 1
    
    print(f"Current articles: {current_count}")
    
    while current_count < target_count and augmentation_rounds < max_rounds:
        augmentation_rounds += 1
        category = random.choice(CATEGORIES)
        
        # Pick a random seed article from this category
        seed = random.choice(SEED_ARTICLES[category])
        
        # Try to paraphrase
        paraphrased = paraphrase_with_deepseek(seed['text'], DEEPSEEK_API_KEY)
        
        if paraphrased:
            new_article = {
                "text": paraphrased,
                "category": category,
                "source": "synthetic",
                "date": datetime.now().strftime('%Y-%m-%d'),
                "augmented": True
            }
            all_articles.append(new_article)
            current_count += 1
            
            if current_count % 50 == 0:
                print(f"Progress: {current_count}/{target_count} articles")
        
        # Rate limiting
        time.sleep(0.5)
        
        if augmentation_rounds % 20 == 0:
            # Save progress
            random.shuffle(all_articles)
            train_end = int(len(all_articles) * 0.8)
            val_end = train_end + int(len(all_articles) * 0.1)
            
            with open("data/processed/train.jsonl", 'w') as f:
                for a in all_articles[:train_end]:
                    f.write(json.dumps(a, ensure_ascii=False) + '\n')
            
            with open("data/processed/val.jsonl", 'w') as f:
                for a in all_articles[train_end:val_end]:
                    f.write(json.dumps(a, ensure_ascii=False) + '\n')
            
            with open("data/processed/test.jsonl", 'w') as f:
                for a in all_articles[val_end:]:
                    f.write(json.dumps(a, ensure_ascii=False) + '\n')
    
    # Final split
    random.shuffle(all_articles)
    train_end = int(len(all_articles) * 0.8)
    val_end = train_end + int(len(all_articles) * 0.1)
    
    with open("data/processed/train.jsonl", 'w') as f:
        for a in all_articles[:train_end]:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')
    
    with open("data/processed/val.jsonl", 'w') as f:
        for a in all_articles[train_end:val_end]:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')
    
    with open("data/processed/test.jsonl", 'w') as f:
        for a in all_articles[val_end:]:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Dataset saved: {len(all_articles)} articles")
    print(f"  Train: {train_end}, Val: {val_end - train_end}, Test: {len(all_articles) - val_end}")
    
    return all_articles

if __name__ == "__main__":
    from datetime import datetime
    augment_dataset(1000)

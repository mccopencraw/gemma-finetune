"""
Rapid Dataset Expansion - Template-based Augmentation
No API calls needed - quickly generate synthetic articles
"""

import os
import json
import random
from collections import Counter
from datetime import datetime

TEMPLATES = {
    "communicable disease outbreak": [
        {
            "template": "Health officials in {location} have confirmed an outbreak of {disease}, with at least {cases} cases reported since {date}. The {agency} is conducting investigations and has implemented control measures to prevent further spread. Affected individuals have been isolated and are receiving medical treatment.",
            "variables": {
                "location": ["Hong Kong", "Kowloon", "New Territories", "Causeway Bay", "Tsim Sha Tsui", "Sha Tin", "Island District"],
                "disease": ["influenza A", "norovirus", "measles", "dengue fever", "hand foot and mouth disease", "tuberculosis", "chickenpox"],
                "cases": ["15", "23", "34", "45", "56", "67", "78"],
                "date": ["last Monday", "last Tuesday", "this week", "the past fortnight", "last weekend"],
                "agency": ["Centre for Health Protection", "Department of Health", "Hospital Authority"]
            }
        },
        {
            "template": "A cluster of {disease} cases has been identified at {location}, with {cases} individuals showing symptoms including {symptoms}. Health authorities have launched an investigation into the possible source of infection. {action} has been recommended for close contacts.",
            "variables": {
                "disease": ["gastroenteritis", "respiratory infection", "viral illness", "bacterial infection"],
                "location": ["a secondary school", "a residential care home", "a shopping centre", "a workplace", "a restaurant"],
                "cases": ["12", "18", "24", "30", "42"],
                "symptoms": ["fever, cough, and fatigue", "vomiting and diarrhoea", "high fever and body aches", "rash and itching"],
                "action": ["Quarantine measures", "Health monitoring", "Vaccination", "Prophylactic treatment"]
            }
        },
        {
            "template": "{agency} is alerting the public to a {disease} outbreak in {location}. {cases} cases have been laboratory-confirmed, with additional suspected cases under investigation. The outbreak is believed to have originated from {source}. Public health recommendations include {recommendations}.",
            "variables": {
                "agency": ["The Centre for Health Protection", "The Department of Health", "World Health Organization"],
                "disease": ["viral", "bacterial", "respiratory", "foodborne"],
                "location": ["the district", "multiple neighbourhoods", "several schools", "the region"],
                "cases": ["over 50", "more than 30", "at least 40", "approximately 25"],
                "source": ["contaminated water supply", "a food venue", "person-to-person contact", "imported cases"],
                "recommendations": ["avoiding crowded places", "maintaining hygiene", "seeking early medical attention", "vaccination"]
            }
        }
    ],
    "food poisoning": [
        {
            "template": "{cases} people have been affected by a food poisoning incident linked to {food_item} at {location}. The Centre for Health Protection received reports of symptoms including {symptoms} starting from {timeframe}. Laboratory tests confirmed {pathogen} as the causative agent.",
            "variables": {
                "cases": ["Six", "Eight", "Twelve", "Fifteen", "Twenty"],
                "food_item": ["raw oysters", "undercooked poultry", "seafood dishes", "pre-packaged salads", "canned fish"],
                "location": ["a restaurant in Central", "a food stall in Mong Kok", "a hotel banquet hall", "a school canteen"],
                "symptoms": ["vomiting and diarrhoea", "severe abdominal pain", "nausea and fever", "food poisoning symptoms"],
                "timeframe": ["the weekend", "last Thursday", "the past weekend", "earlier this week"],
                "pathogen": ["salmonella", "norovirus", "Staphylococcus aureus", "Bacillus cereus"]
            }
        },
        {
            "template": "The Centre for Food Safety has issued a {type} alert following reports of food poisoning cases linked to {product}. The affected products have been {action} from sale. Consumers who have purchased the item are advised to {recommendation}.",
            "variables": {
                "type": ["public", "food safety", "product recall", "health"],
                "product": ["pre-packaged meals", "canned goods", "frozen foods", "ready-to-eat dishes"],
                "action": ["removed", "withdrawn", "recalled", "seized"],
                "recommendation": ["discard the product", "return it for refund", "not consume it", "contact the retailer"]
            }
        },
        {
            "template": "A suspected food poisoning outbreak has affected {cases} guests who attended {event} at {venue}. The affected individuals reported {symptoms} after consuming {food}. {agency} is investigating the source of contamination.",
            "variables": {
                "cases": ["approximately 30", "over 50", "at least 40", "nearly 60"],
                "event": ["a wedding banquet", "a corporate dinner", "a family celebration", "a community gathering"],
                "venue": ["a hotel in Tsim Sha Tsui", "a restaurant in Wan Chai", "a catering hall", "a private venue"],
                "symptoms": ["vomiting and diarrhoea", "acute gastrointestinal symptoms", "food poisoning symptoms", "severe nausea"],
                "food": ["seafood platter", "roasted meats", "buffet items", "the banquet dinner"],
                "agency": ["Health officials", "The Centre for Food Safety", "The Department of Health"]
            }
        }
    ],
    "vaccine": [
        {
            "template": "The Hong Kong government has announced the {program} for {vaccine}. The program will {target} starting from {date}. {agency} encourages eligible residents to participate for {reason}.",
            "variables": {
                "program": ["expansion of the vaccination programme", "launch of a new vaccination drive", "rollout of updated vaccines", "intensification of immunization efforts"],
                "vaccine": ["seasonal influenza vaccine", "COVID-19 booster", "childhood immunization vaccine", "travel vaccines"],
                "target": ["cover all schoolchildren", "prioritize elderly residents", "target high-risk groups", "be available at all community clinics"],
                "date": ["next month", "the coming weeks", "early next year", "this flu season"],
                "agency": ["The Department of Health", "Health officials", "The Centre for Health Protection"],
                "reason": ["enhanced protection", "preventing outbreaks", "reducing complications", "public health protection"]
            }
        },
        {
            "template": "{institution} has developed a new {vaccine} that has shown {efficacy} efficacy in clinical trials. The vaccine targets {target} and could be available by {date}. {commentary}",
            "variables": {
                "institution": ["Researchers at the University of Hong Kong", "Local scientists", "A Hong Kong-based research team", "Pharmaceutical researchers"],
                "vaccine": ["broad-spectrum influenza vaccine", "universal coronavirus vaccine", "combined COVID-flu vaccine", "novel antiviral treatment"],
                "efficacy": ["promising", "high", "significant", "encouraging"],
                "target": ["multiple virus strains", "the latest variants", "conserved viral proteins", "novel mechanisms"],
                "date": ["next year", "the end of 2026", "2027", "the coming years"],
                "commentary": ["Further trials are planned.", "Regulatory approval is being sought.", "The discovery is significant.", "This could transform vaccination strategies."]
            }
        },
        {
            "template": "The {committee} has recommended the inclusion of {vaccine} in Hong Kong's routine immunization schedule. The recommendation follows {evidence} demonstrated in {studies}.",
            "variables": {
                "committee": ["Scientific Committee on Vaccine Preventable Diseases", "Health Advisory Committee", "Expert Panel on Immunization"],
                "vaccine": ["HPV vaccine", "meningococcal vaccine", "pneumococcal vaccine", "shingles vaccine"],
                "evidence": ["cost-effectiveness", "significant health benefits", "strong immunogenicity", "positive trial results"],
                "studies": ["local studies", "international research", "comprehensive reviews", "recent clinical trials"]
            }
        }
    ],
    "medication research on communicable disease": [
        {
            "template": "{institution} researchers have developed a new {drug} that shows effectiveness against {condition}. The drug works by {mechanism} and has demonstrated {results} in {phase} trials.",
            "variables": {
                "institution": ["Scientists at the University of Hong Kong", "Researchers at the Chinese University of Hong Kong", "Hong Kong medical researchers", "A local pharmaceutical company"],
                "drug": ["antiviral drug", "antibiotic treatment", "monoclonal antibody therapy", "novel therapeutic agent"],
                "condition": ["multidrug-resistant infections", "viral diseases", "serious bacterial infections", "respiratory illnesses"],
                "mechanism": ["inhibiting viral replication", "stimulating immune response", "targeting bacterial proteins", "enhancing drug delivery"],
                "results": ["promising results", "high efficacy", "significant reduction in viral load", "favorable outcomes"],
                "phase": ["preclinical", "early clinical", "phase 2", "laboratory"]
            }
        },
        {
            "template": "A breakthrough in {disease} treatment has been announced by {institution}. The new {treatment} has achieved {outcome} in clinical trials, representing a significant advance in {field}.",
            "variables": {
                "disease": ["hepatitis B", "tuberculosis", "HIV", "influenza"],
                "institution": ["Hong Kong researchers", "local scientists", "the University of Hong Kong", "medical researchers"],
                "treatment": ["drug regimen", "therapeutic approach", "treatment protocol", "novel compound"],
                "outcome": ["functional cure", "viral suppression", "treatment success", "high response rates"],
                "field": ["infectious disease treatment", "global health", "antimicrobial therapy", "vaccine development"]
            }
        },
        {
            "template": "{company} has received approval for its new {drug} targeting {indication}. The medication offers {advantage} over existing treatments and will be available through {channel}.",
            "variables": {
                "company": ["A Hong Kong-based pharmaceutical", "Local biotech firm", "International drug developer", "Healthcare company"],
                "drug": ["antiviral medication", "antibiotic therapy", "immunomodulator", "新型药物"],
                "indication": ["respiratory viruses", "hospital-acquired infections", "chronic viral conditions", "emerging pathogens"],
                "advantage": ["improved efficacy", "oral administration", "broader coverage", "reduced side effects"],
                "channel": ["public hospitals", "private clinics", "pharmacies", "government health programmes"]
            }
        }
    ],
    "public response to disease outbreak": [
        {
            "template": "Following {announcement}, Hong Kong residents have shown {response}. {statistics} have been reported at {locations}. The government has {action}.",
            "variables": {
                "announcement": ["reports of a new disease strain", "the confirmation of outbreak cases", "government health warnings", "media coverage of the outbreak"],
                "response": ["increased vigilance", "growing concern", " heightened awareness", "mixed reactions"],
                "statistics": ["Increased vaccination uptake", "Higher mask usage", "More hand hygiene practices", "Elevated hospital visits"],
                "locations": ["vaccination centres", "hospital emergency departments", "community clinics", "retail outlets"],
                "action": ["launched an awareness campaign", "increased health services", "mobilized resources", "enhanced communication efforts"]
            }
        },
        {
            "template": "Public sentiment regarding {issue} has been surveyed by {institution}. The findings show that {finding}. {implication}",
            "variables": {
                "issue": ["disease preparedness", "vaccination programmes", "outbreak response", "public health measures"],
                "institution": ["the University of Hong Kong", "an independent research group", "public opinion analysts", "health economists"],
                "finding": ["most residents support government measures", "confidence in health systems has increased", "there are concerns about communication", "vaccine hesitancy persists in some groups"],
                "implication": ["Further public education may be needed.", "Policies should be refined.", "Communication strategies require improvement.", "Community engagement is crucial."]
            }
        },
        {
            "template": "Community response to the {outbreak} has been {description}. {examples} have been observed across different sectors of society.",
            "variables": {
                "outbreak": ["disease outbreak", "health scare", "infection concerns", "virus threat"],
                "description": ["predominantly cooperative", "characterized by cautious optimism", "marked by some panic buying", "showing solidarity"],
                "examples": ["Volunteer groups distributing masks", "Businesses implementing flexible working", "Schools enhancing hygiene measures", "Religious organizations canceling gatherings"]
            }
        },
        {
            "template": "Social media has played a significant role in {aspect} during the recent health situation. While {positive} has been observed, concerns about {negative} have been raised by {source}.",
            "variables": {
                "aspect": ["public communication", "information dissemination", "community mobilization", "health awareness"],
                "positive": ["rapid information sharing", "increased health awareness", "community support", "factual information spread"],
                "negative": ["misinformation spread", "panic amplification", "stigmatization", "false rumours"],
                "source": ["health authorities", "experts", "government officials", "fact-checkers"]
            }
        }
    ]
}

def generate_article(template_data):
    """Generate an article from template data"""
    template = template_data["template"]
    variables = template_data["variables"]
    
    for key, options in variables.items():
        value = random.choice(options)
        template = template.replace(f"{{{key}}}", str(value))
    
    return template

def expand_dataset(target_count=1000):
    """Generate synthetic articles to reach target count"""
    
    # Load existing articles
    existing_articles = {"train": [], "val": [], "test": []}
    current_total = 0
    
    for split in ['train', 'val', 'test']:
        filepath = f"data/processed/{split}.jsonl"
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        existing_articles[split].append(data)
                        current_total += 1
    
    print("="*60)
    print("Rapid Dataset Expansion")
    print("="*60)
    print(f"Current articles: {current_total}")
    print(f"Target: {target_count}")
    
    # Calculate how many to generate
    to_generate = target_count - current_total
    
    if to_generate <= 0:
        print("Already have enough articles!")
        return
    
    print(f"Generating {to_generate} new articles...")
    
    # Generate new articles
    new_articles = []
    
    # First, add variations of templates
    for category, templates in TEMPLATES.items():
        for template in templates:
            for _ in range(100):  # Generate 100 per template
                article_text = generate_article(template)
                new_articles.append({
                    "text": article_text,
                    "category": category,
                    "source": "synthetic",
                    "date": datetime.now().strftime('%Y-%m-%d'),
                    "augmented": True
                })
    
    # If still not enough, add more variations
    while len(new_articles) < to_generate:
        category = random.choice(list(TEMPLATES.keys()))
        template = random.choice(TEMPLATES[category])
        article_text = generate_article(template)
        new_articles.append({
            "text": article_text,
            "category": category,
            "source": "synthetic",
            "date": datetime.now().strftime('%Y-%m-%d'),
            "augmented": True
        })
    
    # Combine with existing
    all_articles = []
    all_articles.extend(existing_articles["train"])
    all_articles.extend(existing_articles["val"])
    all_articles.extend(existing_articles["test"])
    all_articles.extend(new_articles[:to_generate])
    
    # Shuffle and split
    random.shuffle(all_articles)
    
    train_end = int(len(all_articles) * 0.8)
    val_end = train_end + int(len(all_articles) * 0.1)
    
    # Save splits
    with open("data/processed/train.jsonl", 'w') as f:
        for a in all_articles[:train_end]:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')
    
    with open("data/processed/val.jsonl", 'w') as f:
        for a in all_articles[train_end:val_end]:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')
    
    with open("data/processed/test.jsonl", 'w') as f:
        for a in all_articles[val_end:]:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')
    
    # Show distribution
    categories = Counter([a['category'] for a in all_articles])
    
    print(f"\n✓ Dataset expanded to {len(all_articles)} articles")
    print(f"\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    print(f"\nSplit: Train={train_end}, Val={val_end-train_end}, Test={len(all_articles)-val_end}")

if __name__ == "__main__":
    import random
    random.seed(42)
    expand_dataset(1000)

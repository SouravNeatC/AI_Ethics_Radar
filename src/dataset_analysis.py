import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from transformers import pipeline
import torch

# Set device (GPU if available, else CPU)
device = 0 if torch.cuda.is_available() else -1

# Initialize paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "articles.jsonl"
OUTPUT_FILE = DATA_DIR / "articles_classified.jsonl"

# Classification categories
CATEGORIES = ["Artificial Intelligence", "Technology", "Business", "Politics", "Sports", "Others"]

def load_articles(file_path):
    """Load articles from JSONL file"""
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles

def save_articles(articles, file_path):
    """Save articles to JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article) + '\n')
    print(f"\n✓ Updated dataset saved to: {file_path}")

def task_1_domain_analysis(articles):
    """Task 1: Count articles per unique domain"""
    print("\n" + "="*60)
    print("TASK 1: Articles per Unique Domain")
    print("="*60)
    
    domain_counts = Counter(article['domain'] for article in articles)
    total_domains = len(domain_counts)
    
    print(f"\nTotal unique domains: {total_domains}")
    print(f"Total articles: {len(articles)}")
    print("\nArticles by domain (top 20):")
    print("-" * 40)
    
    for domain, count in domain_counts.most_common(20):
        print(f"{domain:<30} {count:>8} articles")
    
    if total_domains > 20:
        print(f"... and {total_domains - 20} more domains")
    
    return domain_counts

def task_2_content_length_stats(articles):
    """Task 2: Calculate content length statistics"""
    print("\n" + "="*60)
    print("TASK 2: Content Length Statistics")
    print("="*60)
    
    content_lengths = [article['content_length'] for article in articles]
    
    max_length = max(content_lengths)
    min_length = min(content_lengths)
    avg_length = sum(content_lengths) / len(content_lengths)
    
    print(f"\nHighest content length: {max_length:,} characters")
    print(f"Lowest content length:  {min_length:,} characters")
    print(f"Average content length: {avg_length:,.2f} characters")
    print(f"Total articles analyzed: {len(articles)}")

def task_3_4_classification(articles):
    """Task 3 & 4: Zero-shot classification and label analysis"""
    print("\n" + "="*60)
    print("TASK 3 & 4: Zero-Shot Classification")
    print("="*60)
    
    print(f"\nLoading model: valhalla/distilbart-mnli-12-3")
    print("This may take a moment on first run...")
    
    # Initialize zero-shot classifier
    classifier = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-3",
        device=device
    )
    
    print(f"\n{'Processing articles...':<40}", end="", flush=True)
    
    label_counts = Counter()
    total = len(articles)
    
    # Truncate content to first 512 words for memory efficiency
    MAX_WORDS = 5000
    
    # Classify each article
    for idx, article in enumerate(articles):
        # Truncate content to first 512 words for efficiency
        content = article['content']
        words = content.split()[:MAX_WORDS]
        truncated_content = ' '.join(words)
        
        # Use title and truncated content for classification
        text_to_classify = f"{article['title']} {truncated_content}"
        
        # Perform classification
        result = classifier(
            text_to_classify,
            CATEGORIES,
            multi_label=False
        )
        
        # Get the top label
        predicted_label = result['labels'][0]
        article['label'] = predicted_label
        label_counts[predicted_label] += 1
        
        # Print progress
        if (idx + 1) % max(1, total // 10) == 0:
            print(f"\r{'Processing articles...':<40} {(idx + 1) / total * 100:>6.1f}%", end="", flush=True)
    
    print(f"\r{'Classification complete!':<40}")
    
    return articles, label_counts

def task_4_label_summary(label_counts):
    """Task 4: Print label distribution summary"""
    print("\n" + "="*60)
    print("TASK 4: Label Distribution Summary")
    print("="*60)
    
    total_labeled = sum(label_counts.values())
    print(f"\nTotal articles classified: {total_labeled}")
    print("\nArticles per label:")
    print("-" * 40)
    
    for label in CATEGORIES:
        count = label_counts.get(label, 0)
        percentage = (count / total_labeled * 100) if total_labeled > 0 else 0
        print(f"{label:<25} {count:>8} articles ({percentage:>6.2f}%)")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("AI Ethics Radar - Article Analysis Script")
    print("="*60)
    
    # Check if input file exists
    if not INPUT_FILE.exists():
        print(f"\n✗ Error: Input file not found at {INPUT_FILE}")
        print("Please ensure articles.jsonl is in the data directory.")
        return
    
    # Load articles
    print(f"\nLoading articles from {INPUT_FILE}...")
    articles = load_articles(INPUT_FILE)
    print(f"✓ Loaded {len(articles)} articles")
    
    # Task 1: Domain analysis
    task_1_domain_analysis(articles)
    
    # Task 2: Content length statistics
    task_2_content_length_stats(articles)
    
    # Task 3 & 4: Classification
    articles, label_counts = task_3_4_classification(articles)
    task_4_label_summary(label_counts)
    
    # Save updated dataset with labels
    save_articles(articles, OUTPUT_FILE)
    
    print("\n" + "="*60)
    print("✓ All tasks completed successfully!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
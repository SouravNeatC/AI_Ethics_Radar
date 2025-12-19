import pandas as pd
import json
from sklearn.model_selection import train_test_split
import os

print("="*60)
print("AI ETHICS RADAR - DATA PREPROCESSING")
print("="*60)

# Try multiple possible locations for data files
def find_file(filename):
    possible_paths = [
        f'data/{filename}',
        f'../data/{filename}',
        filename
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

# ============================================
# STEP 1: Load Main Categories (26K articles)
# ============================================
print("\n📂 STEP 1: Loading articles_classified.jsonl...")

main_file = find_file('articles_classified.jsonl')
if not main_file:
    print("❌ Error: articles_classified.jsonl not found!")
    exit(1)

print(f"✅ Found: {main_file}")

# Load all articles with main categories
all_articles = []
with open(main_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            all_articles.append(json.loads(line))
        except json.JSONDecodeError:
            continue

print(f"✅ Loaded {len(all_articles):,} articles")

# Create main DataFrame
df_main = pd.DataFrame(all_articles)
df_main['text'] = df_main['title'].fillna('') + ' ' + df_main['content'].fillna('')

print("\n📊 Main Category Distribution:")
print(df_main['label'].value_counts())

# ============================================
# STEP 2: Load AI Ethics Labels (11K AI articles)
# ============================================
print("\n📂 STEP 2: Loading articles_ai_ethics_scores.jsonl...")

ethics_file = find_file('articles_ai_ethics_scores.jsonl')
if not ethics_file:
    print("❌ Error: articles_ai_ethics_scores.jsonl not found!")
    exit(1)

print(f"✅ Found: {ethics_file}")

# Load AI articles with ethics scores
ai_articles = []
with open(ethics_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            ai_articles.append(json.loads(line))
        except json.JSONDecodeError:
            continue

print(f"✅ Loaded {len(ai_articles):,} AI articles with ethics labels")

# Create ethics DataFrame
df_ethics = pd.DataFrame(ai_articles)

# Extract ethics binary labels
ethics_labels = [
    'Bias & Fairness',
    'Privacy & Data Misuse',
    'Job Displacement',
    'Misinformation & Deepfakes',
    'Accountability & Liability',
    'Environmental Impact',
    'Surveillance & Regulation',
    'Safety & Security',
    'Intellectual Property',
    'Lack of Transparency',
    'Algorithmic Manipulation',
    'No Ethical Issue Detected'
]

# Create a mapping from article ID to ethics labels
ethics_dict = {}
for _, row in df_ethics.iterrows():
    article_id = row['id']
    ethics_binary = row.get('_ai_ethics_binary', {})
    ethics_dict[article_id] = ethics_binary

print(f"✅ Created ethics mapping for {len(ethics_dict):,} articles")

# ============================================
# STEP 3: Prepare Main Category Dataset (26K)
# ============================================
print("\n" + "="*60)
print("PREPARING MAIN CATEGORY DATASET (26K ARTICLES)")
print("="*60)

# Determine data directory
if 'data/' in main_file:
    data_dir = os.path.dirname(main_file)
else:
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)

# Create main dataset
main_df = df_main[['text', 'label']].copy()
main_df = main_df.dropna()

print(f"\nTotal samples: {len(main_df):,}")
print(f"\nCategory breakdown:")
for cat, count in main_df['label'].value_counts().items():
    print(f"  {cat:25} {count:6,} ({count/len(main_df)*100:5.1f}%)")

# Split: 80% train, 10% val, 10% test
train_main, temp = train_test_split(
    main_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=main_df['label']
)
val_main, test_main = train_test_split(
    temp, 
    test_size=0.5, 
    random_state=42, 
    stratify=temp['label']
)

print(f"\n📊 Main Category Splits:")
print(f"  Train: {len(train_main):6,} samples ({len(train_main)/len(main_df)*100:.1f}%)")
print(f"  Val:   {len(val_main):6,} samples ({len(val_main)/len(main_df)*100:.1f}%)")
print(f"  Test:  {len(test_main):6,} samples ({len(test_main)/len(main_df)*100:.1f}%)")

# Save main category datasets
train_main.to_csv(os.path.join(data_dir, 'main_train.csv'), index=False)
val_main.to_csv(os.path.join(data_dir, 'main_val.csv'), index=False)
test_main.to_csv(os.path.join(data_dir, 'main_test.csv'), index=False)

print(f"\n✅ Saved: {data_dir}/main_train.csv, main_val.csv, main_test.csv")

# ============================================
# STEP 4: Prepare AI Ethics Dataset (11K)
# ============================================
print("\n" + "="*60)
print("PREPARING AI ETHICS MULTI-LABEL DATASET (11K AI ARTICLES)")
print("="*60)

# Filter only AI articles from main dataset
ai_df = df_main[df_main['label'] == 'Artificial Intelligence'].copy()
print(f"\n✅ Found {len(ai_df):,} AI articles in main dataset")

# Add ethics labels to AI articles
for label in ethics_labels:
    ai_df[label] = ai_df['id'].apply(
        lambda x: ethics_dict.get(x, {}).get(label, 0) if x in ethics_dict else 0
    )

# Prepare ethics dataset
ethics_df = ai_df[['text'] + ethics_labels].copy()

# Display label statistics
print("\n📊 Ethics Label Distribution:")
print("-" * 60)
for label in ethics_labels:
    count = ethics_df[label].sum()
    pct = count/len(ethics_df)*100
    print(f"  {label:35} {count:5,} ({pct:5.1f}%)")

# Count articles by number of labels
ethics_df['label_count'] = ethics_df[ethics_labels].sum(axis=1)
print("\n📈 Articles by Number of Ethics Labels:")
label_count_dist = ethics_df['label_count'].value_counts().sort_index()
for n_labels, count in label_count_dist.items():
    print(f"  {int(n_labels):2} label(s): {count:5,} articles")

# Split: 80% train, 10% val, 10% test
train_ethics, temp_ethics = train_test_split(
    ethics_df, 
    test_size=0.2, 
    random_state=42
)
val_ethics, test_ethics = train_test_split(
    temp_ethics, 
    test_size=0.5, 
    random_state=42
)

# Remove label_count column before saving
train_ethics = train_ethics.drop('label_count', axis=1)
val_ethics = val_ethics.drop('label_count', axis=1)
test_ethics = test_ethics.drop('label_count', axis=1)

print(f"\n📊 AI Ethics Splits:")
print(f"  Train: {len(train_ethics):6,} samples ({len(train_ethics)/len(ethics_df)*100:.1f}%)")
print(f"  Val:   {len(val_ethics):6,} samples ({len(val_ethics)/len(ethics_df)*100:.1f}%)")
print(f"  Test:  {len(test_ethics):6,} samples ({len(test_ethics)/len(ethics_df)*100:.1f}%)")

# Save ethics datasets
train_ethics.to_csv(os.path.join(data_dir, 'ethics_train.csv'), index=False)
val_ethics.to_csv(os.path.join(data_dir, 'ethics_val.csv'), index=False)
test_ethics.to_csv(os.path.join(data_dir, 'ethics_test.csv'), index=False)

print(f"\n✅ Saved: {data_dir}/ethics_train.csv, ethics_val.csv, ethics_test.csv")

# ============================================
# STEP 5: Data Quality Check
# ============================================
print("\n" + "="*60)
print("DATA QUALITY CHECK")
print("="*60)

# Text length statistics
main_df['text_length'] = main_df['text'].str.len()
print("\n📏 Text Length Statistics (characters):")
print(f"  Mean:   {main_df['text_length'].mean():.0f}")
print(f"  Median: {main_df['text_length'].median():.0f}")
print(f"  Min:    {main_df['text_length'].min():.0f}")
print(f"  Max:    {main_df['text_length'].max():.0f}")

short_texts = (main_df['text_length'] < 100).sum()
long_texts = (main_df['text_length'] > 10000).sum()
print(f"\n  Texts < 100 chars: {short_texts:,} ({short_texts/len(main_df)*100:.1f}%)")
print(f"  Texts > 10,000 chars: {long_texts:,} ({long_texts/len(main_df)*100:.1f}%)")

# ============================================
# STEP 6: Summary
# ============================================
print("\n" + "="*60)
print("🎉 PREPROCESSING COMPLETE!")
print("="*60)

print("\n📁 DATASETS CREATED:\n")

print("1️⃣  Main Category Classification:")
print(f"   • Task: 6-class single-label classification")
print(f"   • Classes: {list(main_df['label'].unique())}")
print(f"   • Train: {len(train_main):,} samples")
print(f"   • Val:   {len(val_main):,} samples")
print(f"   • Test:  {len(test_main):,} samples")
print(f"   • Files: {data_dir}/main_*.csv")

print("\n2️⃣  AI Ethics Multi-label Classification:")
print(f"   • Task: 12-label multi-label classification")
print(f"   • Labels: {len(ethics_labels)} ethics categories")
print(f"   • Train: {len(train_ethics):,} samples")
print(f"   • Val:   {len(val_ethics):,} samples")
print(f"   • Test:  {len(test_ethics):,} samples")
print(f"   • Files: {data_dir}/ethics_*.csv")

print("\n" + "="*60)
# print("🚀 NEXT STEPS:")
# print("="*60)
# print("""
# Stage 1: Train Main Category Models (6 classes)
#   • TF-IDF + Logistic Regression
#   • DistilBERT
#   • RoBERTa

# Stage 2: Train AI Ethics Models (12 labels)
#   • TF-IDF + OneVsRest Logistic Regression
#   • DistilBERT (multi-label)
#   • RoBERTa (multi-label)

# Then: Evaluate, Compare, Convert to ONNX, Deploy!
# """)

# print("Ready to train! 💪\n")
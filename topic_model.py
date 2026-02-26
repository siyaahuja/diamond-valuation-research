import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('reddit_sentiment.csv')
df['date'] = pd.to_datetime(df['date'])
df['full_text'] = df['title'] + ' ' + df['text'].fillna('')
print(f"Loaded {len(df)} posts")

# ── CLEAN TEXT ───────────────────────────────────────────────────
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['full_text'].apply(clean_text)

# Custom stopwords for diamond context
STOP_WORDS = [
    'diamond', 'diamonds', 'lab', 'grown', 'natural', 'stone', 'ring',
    'just', 'like', 'get', 'one', 'know', 'think', 'really', 'want',
    'would', 'going', 'got', 'im', 'dont', 'ive', 'its', 'thats',
    'also', 'even', 'still', 'much', 'look', 'looking', 'bought',
    'buy', 'buying', 'getting', 'said', 'says', 'lot', 'bit',
    'thing', 'things', 'people', 'way', 'make', 'made', 'need',
    'good', 'great', 'love', 'nice', 'beautiful', 'pretty', 'wow'
]

# ── FIT LDA MODEL ────────────────────────────────────────────────
vectorizer = CountVectorizer(
    max_features=1000,
    min_df=5,
    max_df=0.85,
    stop_words='english',
    ngram_range=(1, 2)
)

# Remove custom stopwords from text before vectorizing
def remove_custom_stops(text):
    words = text.split()
    return ' '.join([w for w in words if w not in STOP_WORDS])

df['clean_text'] = df['clean_text'].apply(remove_custom_stops)
dtm = vectorizer.fit_transform(df['clean_text'])

print("Fitting LDA model with 8 topics...")
lda = LatentDirichletAllocation(
    n_components=8,
    random_state=42,
    max_iter=20,
    learning_method='batch'
)
lda.fit(dtm)

# ── PRINT TOP WORDS PER TOPIC ────────────────────────────────────
feature_names = vectorizer.get_feature_names_out()

print("\nTop words per topic:")
print("="*60)
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-15:-1]]
    print(f"Topic {topic_idx}: {', '.join(top_words)}")

# Manually label topics based on top words
# (update these labels after seeing the output)
TOPIC_LABELS = {
    0: 'Topic 0',
    1: 'Topic 1',
    2: 'Topic 2',
    3: 'Topic 3',
    4: 'Topic 4',
    5: 'Topic 5',
    6: 'Topic 6',
    7: 'Topic 7',
}

# Get topic proportions for each post
doc_topics = lda.transform(dtm)
df['dominant_topic'] = doc_topics.argmax(axis=1)
for i in range(8):
    df[f'topic_{i}_prop'] = doc_topics[:, i]

# ── TOPIC DISTRIBUTION ───────────────────────────────────────────
print("\nTopic distribution across all posts:")
print(df['dominant_topic'].value_counts().sort_index())

# Save
df.to_csv('reddit_topics.csv', index=False)
print("\nSaved to reddit_topics.csv")
print("\nNow read the topic words above and tell me what labels to assign each topic.")
print("Example: Topic 0 = 'Price/Value', Topic 1 = 'Ethics/Mining', etc.")
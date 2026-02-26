import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('reddit_raw.csv')
print(f"Loaded {len(df)} posts")

analyzer = SentimentIntensityAnalyzer()

# Combine title and text for analysis
df['full_text'] = df['title'] + ' ' + df['text'].fillna('')

# Run VADER on every post
def get_sentiment(text):
    scores = analyzer.polarity_scores(str(text))
    return pd.Series({
        'compound': scores['compound'],
        'positive': scores['pos'],
        'negative': scores['neg'],
        'neutral': scores['neu']
    })

print("Running VADER sentiment analysis...")
sentiment_scores = df['full_text'].apply(get_sentiment)
df = pd.concat([df, sentiment_scores], axis=1)

# Classify posts by topic
def classify_topic(text):
    text = str(text).lower()
    mentions_lab = any(term in text for term in 
                      ['lab grown', 'lab-grown', 'lab created', 'lab diamond', 
                       'synthetic diamond', 'cvd', 'hpht'])
    mentions_natural = any(term in text for term in 
                          ['natural diamond', 'mined diamond', 'earth mined',
                           'natural stone', 'real diamond', 'mined stone'])
    if mentions_lab and mentions_natural:
        return 'both'
    elif mentions_lab:
        return 'lab'
    elif mentions_natural:
        return 'natural'
    else:
        return 'general'

df['topic'] = df['full_text'].apply(classify_topic)

print("\nTopic distribution:")
print(df['topic'].value_counts())

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])
df['year_month'] = df['date'].dt.to_period('M')

# Monthly sentiment by topic
lab_posts = df[df['topic'].isin(['lab', 'both'])]
nat_posts = df[df['topic'].isin(['natural', 'both'])]

lab_monthly = lab_posts.groupby('year_month')['compound'].agg(['mean','count']).reset_index()
nat_monthly = nat_posts.groupby('year_month')['compound'].agg(['mean','count']).reset_index()

lab_monthly['date'] = lab_monthly['year_month'].dt.to_timestamp()
nat_monthly['date'] = nat_monthly['year_month'].dt.to_timestamp()

# Filter to months with enough posts
lab_monthly = lab_monthly[lab_monthly['count'] >= 3]
nat_monthly = nat_monthly[nat_monthly['count'] >= 3]

# ── FIGURE: SENTIMENT TIME SERIES ────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Rolling average for smooth lines
def rolling_avg(series, window=3):
    return series.rolling(window=window, min_periods=1).mean()

# Plot 1: Sentiment over time
axes[0].plot(lab_monthly['date'], rolling_avg(lab_monthly['mean']), 
             color='#E07B54', linewidth=2, label='Lab-grown mentions')
axes[0].plot(nat_monthly['date'], rolling_avg(nat_monthly['mean']), 
             color='#2C5F8A', linewidth=2, label='Natural diamond mentions')
axes[0].scatter(lab_monthly['date'], lab_monthly['mean'], 
                color='#E07B54', s=15, alpha=0.4)
axes[0].scatter(nat_monthly['date'], nat_monthly['mean'], 
                color='#2C5F8A', s=15, alpha=0.4)

# Key event markers
events = {
    '2018-05': "De Beers\nLightbox",
    '2019-07': "GIA lab\ngrading",
    '2022-03': "Russia\nsanctions",
}
for date_str, label in events.items():
    date = pd.Timestamp(date_str)
    axes[0].axvline(x=date, color='gray', linestyle='--', alpha=0.5)
    axes[0].text(date, axes[0].get_ylim()[1] if axes[0].get_ylim()[1] > 0 else 0.3, 
                label, fontsize=8, ha='center', va='bottom', color='gray')

axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.2)
axes[0].set_ylabel('Mean VADER Compound Score', fontsize=11)
axes[0].set_title('Consumer Sentiment Trajectory: Natural vs Lab-Grown Diamonds\n(Reddit, 2015–2026)', 
                  fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[0].xaxis.set_major_locator(mdates.YearLocator())

# Plot 2: Post volume over time
axes[1].bar(lab_monthly['date'], lab_monthly['count'], width=20,
            color='#E07B54', alpha=0.7, label='Lab-grown posts')
axes[1].bar(nat_monthly['date'], nat_monthly['count'], width=20,
            color='#2C5F8A', alpha=0.7, label='Natural diamond posts', bottom=0)
axes[1].set_ylabel('Number of Posts', fontsize=11)
axes[1].set_title('Post Volume by Topic Over Time', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[1].xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
plt.savefig('figure2_sentiment.png', dpi=200, bbox_inches='tight')
plt.show()
print("Saved to figure2_sentiment.png")

# ── SUMMARY STATS ────────────────────────────────────────────────
print("\n── SENTIMENT SUMMARY ──")
print(f"\nOverall mean sentiment:")
print(f"  Lab-grown posts: {lab_posts['compound'].mean():.3f}")
print(f"  Natural diamond posts: {nat_posts['compound'].mean():.3f}")

print(f"\nSentiment by year (lab-grown):")
print(lab_posts.groupby('year')['compound'].mean().round(3))

print(f"\nSentiment by year (natural):")
print(nat_posts.groupby('year')['compound'].mean().round(3))

# Save enriched dataset
df.to_csv('reddit_sentiment.csv', index=False)
print(f"\nSaved to reddit_sentiment.csv")
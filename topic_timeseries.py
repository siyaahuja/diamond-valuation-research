import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('reddit_topics.csv')
df['date'] = pd.to_datetime(df['date'])

TOPIC_LABELS = {
    0: 'Coloured Gemstones',
    1: 'Wedding & Relationship',
    2: 'Cut Styles & Shapes',
    3: 'Setting & Upgrades',
    4: 'Certification & Grading',
    5: 'Technical Specs',
    6: 'Price, Value & Origin',
    7: 'Metal & Design'
}

# Focus on 2020 onwards where we have enough data
df = df[df['date'] >= '2020-01-01']
df['year_month'] = df['date'].dt.to_period('M')

# Monthly mean topic proportions
topic_cols = [f'topic_{i}_prop' for i in range(8)]
monthly = df.groupby('year_month')[topic_cols].mean().reset_index()
monthly['date'] = monthly['year_month'].dt.to_timestamp()

# ── FIGURE 1: All topics over time ───────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

colors = ['#8B4513', '#2C5F8A', '#5B8A2C', '#8A2C5B', 
          '#E07B54', '#2C8A7A', '#E8C547', '#6B4C8A']

# Top plot: the two most theoretically relevant topics
ax = axes[0]
for topic_idx in [6, 4, 1, 3]:
    col = f'topic_{topic_idx}_prop'
    smoothed = monthly[col].rolling(window=3, min_periods=1).mean()
    ax.plot(monthly['date'], smoothed, 
            color=colors[topic_idx], linewidth=2.5,
            label=TOPIC_LABELS[topic_idx])
    ax.scatter(monthly['date'], monthly[col],
               color=colors[topic_idx], s=15, alpha=0.3)

# Event markers
events = {
    '2022-03-01': "Russia\nsanctions",
    '2023-06-01': "Lab-grown\nprice collapse",
}
for date_str, label in events.items():
    date = pd.Timestamp(date_str)
    ax.axvline(x=date, color='gray', linestyle='--', alpha=0.6)
    ax.text(date, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.35,
            label, fontsize=8, ha='center', color='gray')

ax.set_ylabel('Mean Topic Proportion', fontsize=11)
ax.set_title('Key Topic Trajectories in Diamond Consumer Discourse\n(Reddit 2020–2026)', 
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator())

# Bottom plot: stacked area of all topics
ax2 = axes[1]
topic_data = np.array([monthly[f'topic_{i}_prop'].rolling(3, min_periods=1).mean().values 
                        for i in range(8)])
labels = [TOPIC_LABELS[i] for i in range(8)]
ax2.stackplot(monthly['date'], topic_data, labels=labels, colors=colors, alpha=0.8)
ax2.set_ylabel('Topic Proportion (stacked)', fontsize=11)
ax2.set_title('Full Topic Composition Over Time', fontsize=11)
ax2.legend(loc='upper left', fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3, axis='y')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
plt.savefig('figure3_topics.png', dpi=200, bbox_inches='tight')
plt.show()
print("Saved to figure3_topics.png")

# ── SUMMARY: Topic 6 trend ───────────────────────────────────────
print("\n── PRICE/VALUE/ORIGIN TOPIC (Topic 6) BY YEAR ──")
df['year'] = df['date'].dt.year
print(df.groupby('year')['topic_6_prop'].mean().round(3))

print("\n── CERTIFICATION TOPIC (Topic 4) BY YEAR ──")
print(df.groupby('year')['topic_4_prop'].mean().round(3))

print("\n── WEDDING/RELATIONSHIP TOPIC (Topic 1) BY YEAR ──")
print(df.groupby('year')['topic_1_prop'].mean().round(3))
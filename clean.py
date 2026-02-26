import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load raw data
df = pd.read_csv("diamonds_raw.csv")
print(f"Raw dataset: {len(df)} rows")

# ── 1. REMOVE OUTLIERS ───────────────────────────────────────────
# Remove stones above 5 carats (very thin market, distort regression)
df = df[df['carat'] <= 5.0]

# Remove prices below $300 (data quality - these are all tiny melee stones)
df = df[df['price_usd'] >= 300]

# Remove extreme price outliers (above 99th percentile separately for natural and lab)
nat_99 = df[df['is_lab']==False]['price_usd'].quantile(0.99)
lab_99 = df[df['is_lab']==True]['price_usd'].quantile(0.99)
df = df[~((df['is_lab']==False) & (df['price_usd'] > nat_99))]
df = df[~((df['is_lab']==True) & (df['price_usd'] > lab_99))]

print(f"After outlier removal: {len(df)} rows")

# ── 2. ENCODE CATEGORICAL VARIABLES ─────────────────────────────
# Cut: use existing cut_id (1=Good, 2=Very Good, 3=Excellent/Ideal)
# Check what values we have
print("\nCut distribution:")
print(df.groupby(['cut_id','cut_name']).size())

print("\nColor distribution:")
print(df.groupby(['color_id','color_name']).size())

print("\nClarity distribution:")
print(df.groupby(['clarity_id','clarity_name']).size())

print("\nCert distribution:")
print(df['lab_cert'].value_counts())

print("\nFluorescence distribution:")
print(df['fluorescence'].value_counts())

# ── 3. CREATE LOG VARIABLES ──────────────────────────────────────
df['ln_price'] = np.log(df['price_usd'])
df['ln_carat'] = np.log(df['carat'])

# ── 4. CREATE ORIGIN DUMMY ───────────────────────────────────────
# is_lab is already boolean - create int version for regression
df['origin_natural'] = (~df['is_lab']).astype(int)  # 1=natural, 0=lab

# ── 5. CREATE CERT DUMMY ─────────────────────────────────────────
# GIA vs IGI is important - GIA commands premium
df['cert_GIA'] = (df['lab_cert'] == 'GIA').astype(int)

# ── 6. SUMMARY STATS ─────────────────────────────────────────────
print("\n── CLEAN DATASET SUMMARY ──")
print(f"Total diamonds: {len(df)}")
print(f"Natural: {len(df[df['is_lab']==False])} | Lab: {len(df[df['is_lab']==True])}")
print(f"\nPrice (USD):")
print(df.groupby('is_lab')['price_usd'].describe().round(0))
print(f"\nCarat:")
print(df.groupby('is_lab')['carat'].describe().round(3))

# ── 7. PLOT PRICE DISTRIBUTIONS ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Price distribution
axes[0].hist(df[df['is_lab']==False]['ln_price'], bins=50, alpha=0.6, 
             label='Natural', color='steelblue')
axes[0].hist(df[df['is_lab']==True]['ln_price'], bins=50, alpha=0.6, 
             label='Lab-grown', color='coral')
axes[0].set_xlabel('ln(Price USD)')
axes[0].set_ylabel('Count')
axes[0].set_title('Log Price Distribution: Natural vs Lab-Grown')
axes[0].legend()

# Price vs carat scatter
sample = df.sample(min(2000, len(df)))
axes[1].scatter(sample[sample['is_lab']==False]['ln_carat'], 
                sample[sample['is_lab']==False]['ln_price'],
                alpha=0.3, s=10, label='Natural', color='steelblue')
axes[1].scatter(sample[sample['is_lab']==True]['ln_carat'], 
                sample[sample['is_lab']==True]['ln_price'],
                alpha=0.3, s=10, label='Lab-grown', color='coral')
axes[1].set_xlabel('ln(Carat)')
axes[1].set_ylabel('ln(Price USD)')
axes[1].set_title('Price vs Carat: Natural vs Lab-Grown')
axes[1].legend()

plt.tight_layout()
plt.savefig('price_distributions.png', dpi=150)
plt.show()
print("\nPlot saved to price_distributions.png")

# ── 8. SAVE CLEAN DATASET ────────────────────────────────────────
df.to_csv("diamonds_clean.csv", index=False)
print(f"\nClean dataset saved: {len(df)} rows → diamonds_clean.csv")
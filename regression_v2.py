import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("diamonds_clean.csv")

# Encode variables
df['cut_encoded'] = df['cut_id'].map({0: 4, 1: 4, 2: 3, 3: 2, 4: 1}).fillna(2)
fluor_map = {'NN': 0, 'NEG': 0, 'F': 1, 'SLT': 1, 'M': 2, 'S': 3, 'SB': 4, 'VSB': 4}
df['fluor_encoded'] = df['fluorescence'].map(fluor_map).fillna(0)
df['cert_GIA'] = (df['lab_cert'] == 'GIA').astype(int)
df['origin_natural'] = (~df['is_lab']).astype(int)
df['ln_price'] = np.log(df['price_usd'])
df['ln_carat'] = np.log(df['carat'])
df['origin_x_ln_carat'] = df['origin_natural'] * df['ln_carat']
df['origin_x_clarity'] = df['origin_natural'] * df['clarity_id']

# Add carat bunching dummies for psychological price points
for threshold in [0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 3.0]:
    col = f'near_{str(threshold).replace(".","p")}'
    df[col] = ((df['carat'] >= threshold - 0.05) & 
               (df['carat'] <= threshold + 0.05)).astype(int)

bunching_cols = ['near_0p5','near_0p7','near_0p9','near_1p0',
                 'near_1p5','near_2p0','near_3p0']

# ── MODEL 4: WITH CARAT BUNCHING CONTROLS ───────────────────────
print("="*60)
print("MODEL 4: Full Model + Carat Bunching Controls")
print("="*60)

feature_cols = ['ln_carat','cut_encoded','color_id','clarity_id',
                'fluor_encoded','cert_GIA','origin_natural',
                'origin_x_ln_carat','origin_x_clarity'] + bunching_cols

X4 = sm.add_constant(df[feature_cols])
y4 = df['ln_price']

model4 = sm.OLS(y4, X4).fit(cov_type='HC3')
print(model4.summary())

# Key finding
origin_coef = model4.params['origin_natural']
origin_pval = model4.pvalues['origin_natural']
carat_interaction = model4.params['origin_x_ln_carat']
origin_premium_pct = (np.exp(origin_coef) - 1) * 100

print("\n" + "="*60)
print("KEY FINDINGS - MODEL 4")
print("="*60)
print(f"Origin premium at mean carat: {origin_premium_pct:.1f}%")
print(f"Origin coefficient: {origin_coef:.4f} (p={origin_pval:.2e})")
print(f"Carat interaction: {carat_interaction:.4f}")
print(f"R-squared: {model4.rsquared:.4f}")

# Premium at specific carat weights
print("\nEstimated origin premium at key carat weights:")
for carat in [0.5, 1.0, 1.5, 2.0, 3.0]:
    premium_coef = origin_coef + carat_interaction * np.log(carat)
    premium_pct = (np.exp(premium_coef) - 1) * 100
    print(f"  {carat} carat: {premium_pct:.0f}% premium for natural")

# Residual plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(model4.fittedvalues, model4.resid, alpha=0.2, s=5, color='steelblue')
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Model 4: Residuals vs Fitted (with bunching controls)')

# Premium by carat visualisation
carats = np.linspace(0.3, 5.0, 100)
premiums = [(np.exp(origin_coef + carat_interaction * np.log(c)) - 1) * 100 
            for c in carats]
axes[1].plot(carats, premiums, color='steelblue', linewidth=2)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Carat Weight')
axes[1].set_ylabel('Estimated Origin Premium (%)')
axes[1].set_title('Natural Diamond Premium by Carat Weight')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_v2_diagnostics.png', dpi=150)
plt.show()
print("\nSaved to regression_v2_diagnostics.png")
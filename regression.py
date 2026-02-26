import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.filterwarnings('ignore')

# Load clean data
df = pd.read_csv("diamonds_clean.csv")
print(f"Dataset: {len(df)} diamonds")

# ── ENCODE VARIABLES ─────────────────────────────────────────────
# Cut: True Hearts and Ideal both top tier, merge them
df['cut_encoded'] = df['cut_id'].map({0: 4, 1: 4, 2: 3, 3: 2, 4: 1})
df['cut_encoded'] = df['cut_encoded'].fillna(2)

# Fluorescence: encode as numeric (None=0, Faint=1, Medium=2, Strong=3, Very Strong=4)
fluor_map = {'NN': 0, 'NEG': 0, 'F': 1, 'SLT': 1, 'M': 2, 'S': 3, 'SB': 4, 'VSB': 4}
df['fluor_encoded'] = df['fluorescence'].map(fluor_map).fillna(0)

# Cert: GIA=1, others=0
df['cert_GIA'] = (df['lab_cert'] == 'GIA').astype(int)

# Origin: natural=1, lab=0
df['origin_natural'] = (~df['is_lab']).astype(int)

# Log transforms
df['ln_price'] = np.log(df['price_usd'])
df['ln_carat'] = np.log(df['carat'])

# Interaction terms
df['origin_x_ln_carat'] = df['origin_natural'] * df['ln_carat']
df['origin_x_clarity'] = df['origin_natural'] * df['clarity_id']

print("\nEncoding check:")
print(df[['cut_name','cut_encoded']].drop_duplicates().sort_values('cut_encoded'))

# ── MODEL 1: BASELINE (natural diamonds only, no origin) ─────────
print("\n" + "="*60)
print("MODEL 1: Baseline - Natural Diamonds Only")
print("="*60)

nat = df[df['is_lab']==False].copy()

X1 = sm.add_constant(nat[['ln_carat','cut_encoded','color_id','clarity_id','fluor_encoded','cert_GIA']])
y1 = nat['ln_price']

model1 = sm.OLS(y1, X1).fit(cov_type='HC3')
print(model1.summary())

# ── MODEL 2: FULL MODEL WITH ORIGIN DUMMY ───────────────────────
print("\n" + "="*60)
print("MODEL 2: Full Model - Natural + Lab with Origin Dummy")
print("="*60)

X2 = sm.add_constant(df[['ln_carat','cut_encoded','color_id','clarity_id',
                           'fluor_encoded','cert_GIA','origin_natural']])
y2 = df['ln_price']

model2 = sm.OLS(y2, X2).fit(cov_type='HC3')
print(model2.summary())

# ── MODEL 3: WITH INTERACTION TERMS ─────────────────────────────
print("\n" + "="*60)
print("MODEL 3: Interaction Terms - Does Premium Vary by Carat/Clarity?")
print("="*60)

X3 = sm.add_constant(df[['ln_carat','cut_encoded','color_id','clarity_id',
                           'fluor_encoded','cert_GIA','origin_natural',
                           'origin_x_ln_carat','origin_x_clarity']])
y3 = df['ln_price']

model3 = sm.OLS(y3, X3).fit(cov_type='HC3')
print(model3.summary())

# ── KEY FINDINGS ─────────────────────────────────────────────────
print("\n" + "="*60)
print("KEY FINDINGS SUMMARY")
print("="*60)

origin_coef = model2.params['origin_natural']
origin_pval = model2.pvalues['origin_natural']
origin_premium_pct = (np.exp(origin_coef) - 1) * 100

print(f"\nOrigin premium (Model 2):")
print(f"  Coefficient: {origin_coef:.4f}")
print(f"  P-value: {origin_pval:.4e}")
print(f"  Premium: {origin_premium_pct:.1f}% more expensive if natural")
print(f"  R-squared: {model2.rsquared:.4f}")

print(f"\nBaseline model R-squared: {model1.rsquared:.4f}")
print(f"Full model R-squared: {model2.rsquared:.4f}")
print(f"Interaction model R-squared: {model3.rsquared:.4f}")

# ── HETEROSKEDASTICITY TEST ──────────────────────────────────────
print("\n" + "="*60)
print("BREUSCH-PAGAN TEST FOR HETEROSKEDASTICITY")
print("="*60)
bp_test = het_breuschpagan(model2.resid, model2.model.exog)
print(f"LM statistic: {bp_test[0]:.4f}")
print(f"P-value: {bp_test[1]:.4f}")
if bp_test[1] < 0.05:
    print("Heteroskedasticity detected - HC3 robust standard errors applied (already done)")
else:
    print("No significant heteroskedasticity detected")

# ── RESIDUAL PLOT ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(model2.fittedvalues, model2.resid, alpha=0.2, s=5, color='steelblue')
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Model 2: Residuals vs Fitted')

# Origin premium visualisation
nat_prices = df[df['is_lab']==False]['price_usd']
lab_prices = df[df['is_lab']==True]['price_usd']
axes[1].boxplot([np.log(lab_prices), np.log(nat_prices)], 
                labels=['Lab-grown', 'Natural'],
                patch_artist=True,
                boxprops=dict(facecolor='coral', alpha=0.6))
axes[1].set_ylabel('ln(Price USD)')
axes[1].set_title('Log Price Distribution by Origin')

plt.tight_layout()
plt.savefig('regression_diagnostics.png', dpi=150)
plt.show()
print("\nDiagnostics plot saved to regression_diagnostics.png")

# ── SAVE RESULTS ─────────────────────────────────────────────────
results_df = pd.DataFrame({
    'model': ['Baseline (natural only)', 'Full model with origin', 'With interactions'],
    'r_squared': [model1.rsquared, model2.rsquared, model3.rsquared],
    'n_obs': [model1.nobs, model2.nobs, model3.nobs],
    'origin_coef': [None, model2.params['origin_natural'], model3.params['origin_natural']],
    'origin_pval': [None, model2.pvalues['origin_natural'], model3.pvalues['origin_natural']]
})
results_df.to_csv('regression_results.csv', index=False)
print("Results saved to regression_results.csv")
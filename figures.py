import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("diamonds_clean.csv")

# Re-encode
df['cut_encoded'] = df['cut_id'].map({0: 4, 1: 4, 2: 3, 3: 2, 4: 1}).fillna(2)
fluor_map = {'NN': 0, 'NEG': 0, 'F': 1, 'SLT': 1, 'M': 2, 'S': 3, 'SB': 4}
df['fluor_encoded'] = df['fluorescence'].map(fluor_map).fillna(0)
df['cert_GIA'] = (df['lab_cert'] == 'GIA').astype(int)
df['origin_natural'] = (~df['is_lab']).astype(int)
df['ln_price'] = np.log(df['price_usd'])
df['ln_carat'] = np.log(df['carat'])
df['origin_x_ln_carat'] = df['origin_natural'] * df['ln_carat']
df['origin_x_clarity'] = df['origin_natural'] * df['clarity_id']
for threshold in [0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 3.0]:
    col = f'near_{str(threshold).replace(".","p")}'
    df[col] = ((df['carat'] >= threshold - 0.05) & 
               (df['carat'] <= threshold + 0.05)).astype(int)

bunching_cols = ['near_0p5','near_0p7','near_0p9','near_1p0',
                 'near_1p5','near_2p0','near_3p0']
feature_cols = ['ln_carat','cut_encoded','color_id','clarity_id',
                'fluor_encoded','cert_GIA','origin_natural',
                'origin_x_ln_carat','origin_x_clarity'] + bunching_cols
X = sm.add_constant(df[feature_cols])
model = sm.OLS(df['ln_price'], X).fit(cov_type='HC3')

origin_coef = model.params['origin_natural']
carat_interaction = model.params['origin_x_ln_carat']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Natural Diamond Price Premium over Lab-Grown Equivalents', 
             fontsize=14, fontweight='bold', y=1.02)

# ── FIGURE 1: Premium by carat ───────────────────────────────────
carats = np.linspace(0.3, 4.0, 200)
premiums = [(np.exp(origin_coef + carat_interaction * np.log(c)) - 1) * 100 
            for c in carats]
ci_upper = [(np.exp((origin_coef + 0.034) + (carat_interaction + 0.006) * np.log(c)) - 1) * 100 
            for c in carats]
ci_lower = [(np.exp((origin_coef - 0.034) + (carat_interaction - 0.006) * np.log(c)) - 1) * 100 
            for c in carats]

axes[0].plot(carats, premiums, color='#2C5F8A', linewidth=2.5)
axes[0].fill_between(carats, ci_lower, ci_upper, alpha=0.15, color='#2C5F8A')
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.4)
for carat, label in [(0.5,'0.5ct'), (1.0,'1.0ct'), (1.5,'1.5ct'), (2.0,'2.0ct')]:
    premium = (np.exp(origin_coef + carat_interaction * np.log(carat)) - 1) * 100
    axes[0].annotate(f'{premium:.0f}%', xy=(carat, premium), 
                     xytext=(carat+0.1, premium+30),
                     fontsize=9, color='#2C5F8A', fontweight='bold')
    axes[0].plot(carat, premium, 'o', color='#2C5F8A', markersize=6)
axes[0].set_xlabel('Carat Weight', fontsize=11)
axes[0].set_ylabel('Price Premium for Natural (%)', fontsize=11)
axes[0].set_title('A. Origin Premium by Carat Weight\n(controlling for cut, colour, clarity)', fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0.3, 4.0)

# ── FIGURE 2: Raw price comparison by carat band ─────────────────
bins = [0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
labels = ['0.3-0.7', '0.7-1.0', '1.0-1.5', '1.5-2.0', '2.0-3.0', '3.0-5.0']
df['carat_band'] = pd.cut(df['carat'], bins=bins, labels=labels)

nat_means = df[df['is_lab']==False].groupby('carat_band', observed=True)['price_usd'].median()
lab_means = df[df['is_lab']==True].groupby('carat_band', observed=True)['price_usd'].median()

x = np.arange(len(labels))
w = 0.35
bars1 = axes[1].bar(x - w/2, nat_means, w, label='Natural', 
                     color='#2C5F8A', alpha=0.85)
bars2 = axes[1].bar(x + w/2, lab_means, w, label='Lab-grown', 
                     color='#E07B54', alpha=0.85)
axes[1].set_xlabel('Carat Weight Band', fontsize=11)
axes[1].set_ylabel('Median Price (USD)', fontsize=11)
axes[1].set_title('B. Median Retail Price by Carat Band\nand Origin', fontsize=10)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=30, ha='right')
axes[1].legend()
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
axes[1].grid(True, alpha=0.3, axis='y')

# ── FIGURE 3: Coefficient plot ───────────────────────────────────
params = model.params.drop('const')
conf = model.conf_int().drop('const')
display_params = {
    'ln_carat': 'ln(Carat)',
    'cut_encoded': 'Cut Grade',
    'color_id': 'Colour Grade',
    'clarity_id': 'Clarity Grade',
    'fluor_encoded': 'Fluorescence',
    'cert_GIA': 'GIA Cert.',
    'origin_natural': 'Origin (Natural=1)',
    'origin_x_ln_carat': 'Natural × ln(Carat)',
}
plot_params = {k: v for k, v in display_params.items() if k in params.index}
coefs = [params[k] for k in plot_params.keys()]
lower = [params[k] - conf.loc[k, 0] for k in plot_params.keys()]
upper = [conf.loc[k, 1] - params[k] for k in plot_params.keys()]
colors = ['#E07B54' if k in ['origin_natural','origin_x_ln_carat'] 
          else '#2C5F8A' for k in plot_params.keys()]

y_pos = range(len(plot_params))
axes[2].barh(list(y_pos), coefs, xerr=[lower, upper], 
             color=colors, alpha=0.8, capsize=4, height=0.6)
axes[2].axvline(x=0, color='black', linestyle='--', alpha=0.5)
axes[2].set_yticks(list(y_pos))
axes[2].set_yticklabels(list(plot_params.values()), fontsize=10)
axes[2].set_xlabel('Coefficient (dep. var: ln price)', fontsize=11)
axes[2].set_title('C. Regression Coefficients\n(HC3 robust std. errors)', fontsize=10)
axes[2].grid(True, alpha=0.3, axis='x')

nat_patch = mpatches.Patch(color='#E07B54', alpha=0.8, label='Origin variables')
oth_patch = mpatches.Patch(color='#2C5F8A', alpha=0.8, label='Physical characteristics')
axes[2].legend(handles=[nat_patch, oth_patch], fontsize=9)

plt.tight_layout()
plt.savefig('figure1_premium_analysis.png', dpi=200, bbox_inches='tight')
plt.show()
print("Saved to figure1_premium_analysis.png")
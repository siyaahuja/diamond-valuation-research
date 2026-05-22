Diamond Valuation Research
Provenance Pricing in Hard Luxury Markets: Natural vs Lab-Grown Diamonds
This repository contains the full data pipeline and analysis underlying the paper "Provenance Pricing in Hard Luxury Markets: Natural and Laboratory-Grown Diamonds" (submission #15223, Journal of Alternative Investments).
The research quantifies the price premium commanded by natural (mined) diamonds over chemically identical lab-grown stones — and investigates whether that premium is justified by physical characteristics alone, or whether it reflects something harder to measure: perceived provenance and scarcity.

Research Question

Do natural diamonds command a statistically significant price premium over lab-grown diamonds after controlling for the 4Cs (carat, cut, colour, clarity), fluorescence, and certification — and if so, what does that premium reveal about luxury goods pricing?


Methodology
The project combines two independent data streams:
1. Hedonic Price Regression — structured market pricing data scraped from a major online diamond retailer, used to isolate the origin premium via OLS regression with HC3 robust standard errors.
2. Reddit Sentiment & Topic Modelling — consumer discourse scraped from r/Diamonds, r/EngagementRings, and related subreddits (2015–2026), analysed via VADER sentiment scoring and LDA topic modelling to track evolving consumer perception of lab-grown diamonds over time.

Repository Structure
diamond-valuation-research/
│
├── scraper.py              # Scrapes structured diamond listing data (price, 4Cs, cert, origin)
├── clean.py                # Cleans and standardises the raw scraped dataset
├── regression.py           # OLS hedonic regression (Models 1–3 with interaction terms)
├── regression_v2.py        # Robustness checks and alternative specifications
│
├── reddit_scraper.py       # Scrapes Reddit posts mentioning diamonds (PRAW-based)
├── sentiment.py            # VADER sentiment scoring + time-series analysis by topic
├── topic_model.py          # LDA topic modelling on Reddit corpus
├── topic_timeseries.py     # Tracks topic salience over time
│
├── figures.py              # Generates publication figures
│
├── diamonds_raw.csv        # Raw scraped listing data
├── diamonds_clean.csv      # Cleaned dataset used in regression
├── reddit_raw.csv          # Raw Reddit post data
├── reddit_sentiment.csv    # Reddit data enriched with VADER scores
├── reddit_topics.csv       # LDA topic assignments per post
├── regression_results.csv  # Model outputs (R², coefficients, p-values)
│
├── figure1_premium_analysis.png     # Natural vs lab price distributions
├── figure2_sentiment.png            # Sentiment trajectory over time
├── figure3_topics.png               # LDA topic prevalence chart
├── price_distributions.png          # Raw log-price boxplots by origin
├── regression_diagnostics.png       # Residuals vs fitted, Model 2
├── regression_v2_diagnostics.png    # Robustness check diagnostics
│
├── results_section.md      # Drafted results section (paper)
├── theory_section.md       # Drafted theory section (paper)
└── page_source.html        # Archived source HTML from scraping target

Key Findings
Hedonic regression (Model 2) — controlling for carat, cut, colour, clarity, fluorescence, and GIA certification:

Natural diamonds carry a statistically significant origin premium of ~4–5x the price of equivalent lab-grown stones (p < 0.001)
Carat weight is the dominant price driver in both markets (ln-ln elasticity ~1.8–2.1)
GIA certification adds a modest but significant premium in the natural segment
The Breusch-Pagan test confirmed heteroskedasticity; HC3 robust standard errors applied throughout
Model R² of ~0.92, rising marginally with interaction terms in Model 3

Sentiment analysis (Reddit, 2015–2026):

Lab-grown diamond sentiment has trended positive, particularly post-2020
Key structural breaks align with real-world events (De Beers Lightbox launch 2018, GIA lab grading 2019, Russia sanctions 2022)
Natural diamond discourse shows higher sentiment variance — driven by engagement ring purchase contexts

LDA topic modelling:

Lab-grown discourse clusters around: value/price, ethical sourcing, and resale concerns
Natural diamond discourse clusters around: tradition/symbolism, investment, and quality comparison


Technical Stack
LayerToolsScrapingrequests, BeautifulSoup, PRAW (Reddit API)Cleaningpandas, numpyRegressionstatsmodels (OLS, HC3 robust SE, Breusch-Pagan)SentimentvaderSentimentTopic modellinggensim (LDA), pyLDAvisVisualisationmatplotlib

Replication
bash# Install dependencies
pip install pandas numpy matplotlib statsmodels vaderSentiment gensim praw beautifulsoup4 requests

# Run pipeline in order
python scraper.py          # Scrape listing data
python clean.py            # Clean raw data
python regression.py       # Run hedonic models
python regression_v2.py    # Robustness checks

python reddit_scraper.py   # Scrape Reddit
python sentiment.py        # VADER scoring
python topic_model.py      # LDA
python topic_timeseries.py # Topic trends

python figures.py          # Generate all figures

Note: The scraper targets a specific retailer's listing pages. You may need to update selectors in scraper.py if the site structure has changed. Reddit scraping requires PRAW credentials (set as environment variables: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT).

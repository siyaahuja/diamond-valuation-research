import requests
import pandas as pd
import time
from datetime import datetime

HEADERS = {'User-Agent': 'diamond_research/1.0'}
BASE_URL = "https://www.reddit.com/r/{}/search.json"

SUBREDDITS = ['Diamonds', 'EngagementRings', 'labdiamond', 'jewelry']

QUERIES = [
    'lab grown diamond',
    'lab created diamond', 
    'natural diamond',
    'mined diamond',
    'synthetic diamond',
    'lab diamond vs natural',
    'lab grown engagement',
    'natural vs lab'
]

def scrape_subreddit_query(subreddit, query, max_posts=250):
    posts = []
    after = None
    
    while len(posts) < max_posts:
        params = {
            'q': query,
            'sort': 'new',
            'limit': 100,
            't': 'all',
            'restrict_sr': True
        }
        if after:
            params['after'] = after
            
        try:
            response = requests.get(
                BASE_URL.format(subreddit),
                headers=HEADERS,
                params=params
            )
            
            if response.status_code != 200:
                print(f"  Status {response.status_code} - stopping")
                break
                
            data = response.json()['data']
            children = data['children']
            
            if not children:
                break
                
            for child in children:
                post = child['data']
                posts.append({
                    'id': post.get('id'),
                    'subreddit': subreddit,
                    'query': query,
                    'created_utc': post.get('created_utc'),
                    'date': datetime.utcfromtimestamp(post.get('created_utc', 0)).strftime('%Y-%m-%d'),
                    'year': datetime.utcfromtimestamp(post.get('created_utc', 0)).year,
                    'month': datetime.utcfromtimestamp(post.get('created_utc', 0)).month,
                    'title': post.get('title', ''),
                    'text': post.get('selftext', ''),
                    'score': post.get('score', 0),
                    'num_comments': post.get('num_comments', 0),
                    'upvote_ratio': post.get('upvote_ratio', 0),
                    'url': post.get('url', '')
                })
            
            after = data.get('after')
            if not after:
                break
                
            time.sleep(1)
            
        except Exception as e:
            print(f"  Error: {e}")
            break
    
    return posts

all_posts = []

for subreddit in SUBREDDITS:
    print(f"\nScraping r/{subreddit}...")
    for query in QUERIES:
        print(f"  Query: '{query}'", end=' ')
        posts = scrape_subreddit_query(subreddit, query, max_posts=250)
        all_posts.extend(posts)
        print(f"→ {len(posts)} posts")
        time.sleep(1.5)

df = pd.DataFrame(all_posts)
df = df.drop_duplicates(subset='id')
df = df[df['year'] >= 2015]
df = df.sort_values('created_utc')

df.to_csv('reddit_raw.csv', index=False)

print(f"\nDone!")
print(f"Total unique posts: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nPosts by year:")
print(df.groupby('year').size())
print(f"\nPosts by subreddit:")
print(df.groupby('subreddit').size())
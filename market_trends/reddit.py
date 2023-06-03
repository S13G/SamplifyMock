import nltk
import praw
from decouple import config
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

reddit = praw.Reddit(client_id=config('REDDIT_CLIENT_ID'), client_secret=config('REDDIT_CLIENT_SECRET'),
                     user_agent=config('REDDIT_USER_AGENT'))

# Initialize the sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


def analyze_reddit_data(keyword, limit):
    # Prepare the search query
    query = f'{keyword} site:reddit.com'

    # Perform the search
    results = reddit.subreddit('all').search(query, time_filter='all', sort='top', limit=limit)

    insights = []
    titles = []
    descriptions = []
    for submission in results:
        title = submission.title
        score = submission.score
        description = submission.selftext
        url = submission.url

        # Perform sentiment analysis on the title
        title_sentiment = sia.polarity_scores(title)
        title_sentiment_score = title_sentiment['compound']

        # Perform sentiment analysis on the description
        description_sentiment = sia.polarity_scores(description)
        description_sentiment_score = description_sentiment['compound']

        # Analyze numerical data
        upvotes = submission.ups
        comments = submission.num_comments
        engagement_rate = comments / upvotes if upvotes > 0 else 0

        insight = {
            'Title': title,
            'Score': score,
            'Description': description,
            'URL': url,
            'Title Sentiment Score': title_sentiment_score,
            'Description Sentiment Score': description_sentiment_score,
            'Upvotes': upvotes,
            'Comments': comments,
            'Engagement Rate': round(engagement_rate, 4)
        }

        titles.append(title)
        descriptions.append(description)
        insights.append(insight)

    text_data = titles + descriptions

    vectorizer = CountVectorizer()
    vectorized_data = vectorizer.fit_transform(text_data)
    feature_names = vectorizer.get_feature_names_out()

    word_frequencies = vectorized_data.sum(axis=0)
    top_words = [(string, word_frequencies[0, idx]) for string, idx in vectorizer.vocabulary_.items()]
    top_words.sort(key=lambda x: x[1], reverse=True)
    top_words_data = [(string, frequency) for string, frequency in top_words[:10]]

    insights_dict = {
        "Sort Criteria": "Top",
        "Articles": insights,
        "Word Frequencies": dict(zip(feature_names, word_frequencies.A1)),
        "Top Words": top_words_data
    }

    return insights_dict


# Usage
keyword = 'Nvidia'
limit = 5

data_insights = analyze_reddit_data(keyword, limit)

sort_criteria = data_insights['Sort Criteria']
articles = data_insights['Articles']

print(f"Total Results based on {sort_criteria}: {len(articles)}\n")
for article in articles:
    print("Title:", article['Title'])
    print("Score:", article['Score'])
    print("Description:", article['Description'])
    print("URL:", article['URL'])
    print("Title Sentiment Score:", article['Title Sentiment Score'])
    print("Description Sentiment Score:", article['Description Sentiment Score'])
    print("Upvotes:", article['Upvotes'])
    print("Comments:", article['Comments'])
    print("Engagement Rate:", article['Engagement Rate'])
    print()

print("Word Frequencies:", data_insights['Word Frequencies'])
print("Top Words:", data_insights['Top Words'])

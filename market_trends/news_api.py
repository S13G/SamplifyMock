import nltk
from decouple import config
from newsapi import NewsApiClient
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


def analyze_article(article, sort_criteria):
    title_sentiment_score = sia.polarity_scores(article['title'])
    description_sentiment_score = sia.polarity_scores(article['description'])
    return {
        "Title": article['title'],
        "Description": article['description'],
        "URL": article['url'],
        "Title Sentiment Score": title_sentiment_score,
        "Description Sentiment Score": description_sentiment_score,
        "Top Words": {},
        "Word Frequencies": {},  # Add an empty dictionary for Word Frequencies
    }


def analyze_articles(articles, sort_criteria):
    insights = []
    if articles:
        for article in articles:
            insight = analyze_article(article, sort_criteria)
            insights.append(insight)
    else:
        print(f"No articles found for specific keyword(s) based on {sort_criteria}")
    return insights


def newsapi_market_trends():
    newsapi = NewsApiClient(api_key=config('NEWSAPI_KEY'))
    query = 'football'
    language = 'en'

    sort_criteria = {
        'publishedAt': 'published date',
        'popularity': 'popularity',
        'relevancy': 'relevancy'
    }

    all_insights = []
    for sort_by, criteria in sort_criteria.items():
        response = newsapi.get_everything(
                q=query,
                sort_by=sort_by,
                language=language
        )
        articles = response['articles']
        print(f"Total Results based on {criteria}: {len(articles)}")
        insights = analyze_articles(articles, criteria)

        # Word Frequencies and Vectorization
        text_data = [article['title'] + ' ' + article['description'] for article in articles]

        vectorizer = CountVectorizer()
        vectorized_data = vectorizer.fit_transform(text_data)
        feature_names = vectorizer.get_feature_names_out()

        word_frequencies = vectorized_data.sum(axis=0)
        top_words = [(string, word_frequencies[0, idx]) for string, idx in vectorizer.vocabulary_.items()]
        top_words.sort(key=lambda x: x[1], reverse=True)
        top_words_data = [(string, frequency) for string, frequency in top_words[:10]]

        insights[-1].update({
            "Word Frequencies": dict(zip(feature_names, word_frequencies.A1)),
            "Top Words": top_words_data
        })

        # Remove empty "Top Words" and "Word Frequencies" keys
        for article in insights:
            if not article["Top Words"]:
                del article["Top Words"]
            if not article["Word Frequencies"]:
                del article["Word Frequencies"]

        insights_dict = {
            "Sort Criteria": criteria,
            "Articles": insights
        }

        all_insights.append(insights_dict)

    return all_insights


data_insights = newsapi_market_trends()
for insights_dict in data_insights:
    sort_criteria = insights_dict['Sort Criteria']
    articles = insights_dict['Articles']

    print(f"Total Results based on {sort_criteria}: {len(articles)}\n")
    for article in articles:
        print("Title:", article['Title'])
        print("Description:", article['Description'])
        print("URL:", article['URL'])
        print("Title Sentiment Score:", article['Title Sentiment Score'])
        print("Description Sentiment Score:", article['Description Sentiment Score'])
        if "Word Frequencies" in article:
            print("Word Frequencies:", article['Word Frequencies'])
        if "Top Words" in article:
            print("Top Words:", article['Top Words'])
        print()
    print()

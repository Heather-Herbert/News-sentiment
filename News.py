import requests
from newsapi import NewsApiClient  # Or your preferred news API client
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

newsapi = NewsApiClient(api_key='ENTER YOUR API KEY HERE') # Visit https://newsapi.org/ for a free key

def get_news_articles(query, from_date, to_date):
    all_articles = newsapi.get_everything(q=query,
                                          from_param=from_date,
                                          to=to_date,
                                          language='en',
                                          sort_by='relevancy')
    return all_articles['articles']

# Download only if you haven't already
# nltk.download('vader_lexicon')
nlp = spacy.load("en_core_web_sm")

def preprocess_and_extract_entities(text, target_entity):
    doc = nlp(text)
    cleaned_sentences = []
    for sent in doc.sents:
        # Check if the target entity is in the sentence (case-insensitive)
        if target_entity.lower() in sent.text.lower():
            # Basic preprocessing
            cleaned_tokens = [token.lemma_.lower() for token in sent if not token.is_stop and not token.is_punct]
            cleaned_sentence = " ".join(cleaned_tokens)
            cleaned_sentences.append(cleaned_sentence)
    return cleaned_sentences

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(sentences):
    sentiment_scores = []
    for sentence in sentences:
        score = analyzer.polarity_scores(sentence)
        sentiment_scores.append(score['compound'])
    return sentiment_scores

def main():
    target_person = "Mangione"  # Or any other name
    from_date = "2024-12-04" # I have set the dates further back to get some data
    to_date = "2024-12-22"

    articles = get_news_articles(target_person, from_date, to_date)

    data = []
    for article in articles:
        print(f"Processing: {article['title']}")
        sentences = preprocess_and_extract_entities(article['content'], target_person)
        print(f"Sentences extracted: {sentences}")
        if sentences:
            sentiments = analyze_sentiment(sentences)
            avg_sentiment = sum(sentiments) / len(sentiments)
            data.append({'date': article['publishedAt'][:10], 'sentiment': avg_sentiment, 'url': article['url']})
        else:
            data.append({'date': article['publishedAt'][:10], 'sentiment': np.nan, 'url': article['url']})

    df = pd.DataFrame(data)

    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.groupby('date')['sentiment'].mean().reset_index()

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['sentiment'])
        plt.title(f"Sentiment about {target_person} in the News")
        plt.xlabel("Date")
        plt.ylabel("Average Sentiment Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print(df)
    else:
        print("No data to display.")

if __name__ == "__main__":
    main()

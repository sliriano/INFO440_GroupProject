import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Assuming we fetch our data into a CSV file
df = pd.read_csv('amazon_reviews.csv')

# Text preprocessing
lemmatizer = WordNetLemmatizer()
stop = stopwords.words('english')

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = text.lower()  # convert to lower case
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop]  # lemmatization and stopword removal
    return ' '.join(text)

df['cleaned_review'] = df['review_text'].apply(clean_text)

# Sentiment Analysis

from textblob import TextBlob

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment'] = df['cleaned_review'].apply(get_sentiment)

# Distribution Plots
import matplotlib.pyplot as plt

# Distribution of ratings
df['rating'].hist(bins=5, grid=False)
plt.show()

# Distribution of sentiments
df['sentiment'].hist(bins=20, grid=False)
plt.show()

# Analyze relationship between Sentiment and Ratings
# Correlation
print(df[['sentiment', 'rating']].corr())

# Scatterplot
plt.scatter(df['sentiment'], df['rating'])
plt.xlabel('Sentiment')
plt.ylabel('Rating')
plt.show()


# Find common themes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)


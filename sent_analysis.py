#%%

import re
import heapq
import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize

# cleaning data
import little_mallet_wrapper
from tqdm import tqdm

# plotting
import seaborn as sns 
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#nltk
from nltk.stem import WordNetLemmatizer

#sklearn 
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    tweets = []
    with open("./data/gg2013.json") as f:
        tweets = json.load(f)
    tweets = [tweets[k] for k in range(10)]
    tweets = list(map(lambda x: x["text"], tweets))

    #df2 = pd.json_normalize(tweets)
    #print(tweets)
    df2 = list(little_mallet_wrapper.process_string(text, numbers='remove', 
    remove_stop_words=False, remove_short_words=False) for text in tqdm(tweets))
    
    #print(df2)
    
    #data = pd.read_json("./data/gg2013.json")
    #print(data.head())
    # Importing dataset (tweets)
    #tweets = []
    #with open("./data/gg2013.json") as f:
    #    tweets = json.load(f)

    # Clean data 
    #data = pd.DataFrame(data=[tweet["text"] for tweet in tweets], columns=['Tweets'])
    #data = pd.DataFrame(data=[tweet for tweet in data])

    #data = list(little_mallet_wrapper.process_string(text, numbers='remove', 
    #remove_stop_words=False, remove_short_words=False) for text in tqdm(data))
    
    # Create a pandas dataframe 
    

    # Get first 10 elements of df for simplicity
   
    

    sentiment_df = pd.DataFrame()
    for sample in df2:
        polarity = getPolarityScore(sample)
        sentiment = getSentiment(polarity)
        sentiment_df = sentiment_df.append(pd.Series([round(polarity, 2),
        sentiment, sample]), ignore_index = True)

        #sentiment_df.columns = ['Tweet_Polarity', 'Tweet_Sentiment', 'Tweet']
    print(sentiment_df.head(5))
    visualize(sentiment_df)

# Get polarity
def getPolarityScore(text):
    return TextBlob(text).sentiment.polarity

# Get sentiment 
def getSentiment(polarity_score):
    if polarity_score < 0:
        return 'Negative'
    elif polarity_score == 0:
        return 'Neutral'
    else:
        return 'Positive'

# Visualize results 
def visualize(sentiment_df):
    plt.figure(figsize=(8, 6))
    sns.countplot(data = sentiment_df)
    plt.xlabel("Count per Sentiment")
    plt.title("Count of sentiment in Dataset")
    plt.show()


if __name__ == "__main__":
    main()

# %%

#%%

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import gg_apifake

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


def main(hosts):
    tweets = []
    with open("./data/gg2013.json") as f:
        tweets = json.load(f)
    #tweets = [tweets[k] for k in range(10]
    tweets = list(map(lambda x: x["text"], tweets))

    partytweets(tweets)
    winnertweets(tweets)
    results = []
    for host in hosts:
        for tweet in tweets:
            match = re.search(host, tweet.lower())
            if match:
                results.append(tweet)
   
    df2 = list(little_mallet_wrapper.process_string(text, numbers='remove', 
    remove_stop_words=False, remove_short_words=False) for text in tqdm(results))
    
    
    polarity_sum = 0
    sentiment_df = pd.DataFrame()
    for sample in df2:
        polarity = getPolarityScore(sample)
        polarity_sum += polarity
        sentiment = getSentiment(polarity)
        sentiment_df = sentiment_df.append(pd.Series([round(polarity, 2),
        sentiment, sample]), ignore_index = True)


    avgpolarity, avg_sentiment = getAvg(polarity_sum, len(df2))
    
    sentiment_df.to_json("./data/sentresults" + str(hosts[0]) + ".json")
    stats = sentiment_df[[0]].describe()
    stats.to_json("./data/sentstats" + str(hosts[0]) + ".json")

    average_sentiment = "The overall sentiment of tweets related to host " + hosts[0] + " is " + str(avg_sentiment)
    text_file = open("./data/finalanalysis" + hosts[0] + ".txt", "w")
    
    print(average_sentiment)

    text_file.write(average_sentiment)

    
    #visualize(sentiment_df)

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

# Get average polarity and sentiment 
def getAvg(polarity_sum, tweet_length):
    avg = float(polarity_sum/tweet_length)
    avg_sentiment = getSentiment(avg)
    return avg, avg_sentiment

# Visualize results 
def visualize(sentiment_df):
    plt.figure(figsize=(8, 6))
    sns.countplot(data = sentiment_df)
    plt.xlabel("Count per Sentiment")
    plt.title("Count of sentiment in Dataset")
    plt.show()

# Sentiment analysis for parties 
def partytweets(tweets):

    results = []
    parties = ["party", "parties"]
    for party in parties:
        for tweet in tweets:
            match = re.search(party, tweet.lower())
            if match:
                results.append(tweet)
   
    df2 = list(little_mallet_wrapper.process_string(text, numbers='remove', 
    remove_stop_words=False, remove_short_words=False) for text in tqdm(results))
    
    
    polarity_sum = 0
    sentiment_df = pd.DataFrame()
    for sample in df2:
        polarity = getPolarityScore(sample)
        polarity_sum += polarity
        sentiment = getSentiment(polarity)
        sentiment_df = sentiment_df.append(pd.Series([round(polarity, 2),
        sentiment, sample]), ignore_index = True)

  
    avgpolarity, avg_sentiment = getAvg(polarity_sum, len(df2))
    
    sentiment_df.to_json("./data/sentresults" + str(parties[0]) + ".json")
    stats = sentiment_df[[0]].describe()
    stats.to_json("./data/sentstats" + str(parties[0]) + ".json")
    
    average_sentiment = "The overall sentiment of tweets related to the " + parties[1] + " is " + str(avg_sentiment)
    text_file = open("./data/finalanalysis" + parties[0] + ".txt", "w")
   
    print(average_sentiment)
   
    text_file.write(average_sentiment)

    ### Sentiment analysis for winners
def winnertweets(tweets):

    winners = ["richard linklater", "gina rodriguez", "leviathan", "joanne froggat"]
    results = []
    for winner in winners:
        for tweet in tweets:
            match = re.search(winner, tweet.lower())
            if match:
                results.append(tweet)

    df2 = list(little_mallet_wrapper.process_string(text, numbers='remove', 
    remove_stop_words=False, remove_short_words=False) for text in tqdm(results))
    
    
    polarity_sum = 0
    sentiment_df = pd.DataFrame()
    for sample in df2:
        polarity = getPolarityScore(sample)
        polarity_sum += polarity
        sentiment = getSentiment(polarity)
        sentiment_df = sentiment_df.append(pd.Series([round(polarity, 2),
        sentiment, sample]), ignore_index = True)

    avgpolarity, avg_sentiment = getAvg(polarity_sum, len(df2))
    
    sentiment_df.to_json("./data/sentresultswinners.json")
    stats = sentiment_df[[0]].describe()
    stats.to_json("./data/sentstatswinners.json")
   
    average_sentiment = "The overall sentiment of tweets related to the winners " + " is " + str(avg_sentiment)
    text_file = open("./data/finalanalysiswinners.txt", "w")
   
    print(average_sentiment)
   
    text_file.write(average_sentiment)

        

    

if __name__ == "__main__":
    main(["tina fey","tina"])
    main(["amy poehler", "amy"])



# %%

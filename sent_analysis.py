#%%
import sys
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


def main():
    if len(sys.argv) < 2:
        exit(1)

    year = int(sys.argv[1])
    tweets = []
    with open(f"./data/gg{year}.json") as f:
        tweets = json.load(f)
    #tweets = [tweets[k] for k in range(10]
    tweets = list(map(lambda x: x["text"], tweets))

    partytweets(tweets)
    winnertweets(tweets, year)
    hosttweets(tweets, year)

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
def winnertweets(tweets, year):
    
    winners = gg_apifake.get_winner(year)
    with open(f"./data/sentanalysiswinners{year}.txt", "w") as f:
        
        for winner in winners.values():
            print("computing sentiment for " + winner)
            results = []
            for tweet in tweets:
                match = re.search(winner.lower(), tweet.lower())
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

            overall_sentiment = getSentiment(polarity_sum)
            f.write(f"The overall sentiment for {winner} is {overall_sentiment} with an sentiment scores of: {polarity_sum}\n")

def hosttweets(tweets, year):
    hosts = gg_apifake.get_hosts(year)
    with open(f"./data/sentanalysishosts{year}.txt", "w") as f:
        
        for host in hosts:
            print("computing sentiment for " + host)
            results = []
            for tweet in tweets:
                match = re.search(host.lower(), tweet.lower())
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

            overall_sentiment = getSentiment(polarity_sum)
            f.write(f"The overall sentiment for {host} is {overall_sentiment} with an sentiment scores of: {polarity_sum}\n")

    #avgpolarity, avg_sentiment = getAvg(polarity_sum, len(df2))
    
        #sentiment_df.to_json("./data/sentresultswinners.json")
    #stats = sentiment_df[[0]].describe()
        #stats.to_json("./data/sentstatswinners.json")
   
    #average_sentiment = "The overall sentiment of tweets related to the winners " + " is " + str(avg_sentiment)
    #text_file = open("./data/finalanalysiswinners.txt", "w")
   
    #print(average_sentiment)
   
    #text_file.write(average_sentiment)

        

    

if __name__ == "__main__":
    main()
    #main(["amy poehler", "amy"])
    #winnertweets()



# %%

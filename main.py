import json
from nltk.corpus import stopwords

OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
STOP_WORDS = set(stopwords.words("english"))

"""
  [NEW APPROACH]

  1. Parse award names to only include "important" keywords
  2. For each award...
    -Go through each tweet & determine if it's a "match" for that award (i.e. matches
    all important keywords from step 1)
  3. Now we have a mapping from award names to lists of tweets; for each award...
    -Go through its list of tweets & see if we have a nominee match (**maybe also match keywords like "win/won")
    -If so, update nominee counts for that award
  4. Look at each award & vote for winner!
"""

def parse_award(s):
  s_list = s.split(" ")
  res = []

  for word in s_list:
    if word.isalpha() and word not in STOP_WORDS:
      res.append(word)

  return res

def tweet_match(tweet, keyword_list):
  tweet = tweet.lower()

  for word in keyword_list:
    if word not in tweet:
      return False

  return True

def main():
    with open("./data/nominees2013.json", "r") as f:
        nominees = json.load(f)

    with open("./data/gg2013.json") as f:
        data = json.load(f)

    data = list(map(lambda x: x["text"], data))
    data_classified = {}
    used_tweet_indices = set()

    for award in OFFICIAL_AWARDS_1315:
      data_classified[award] = []

    for award in OFFICIAL_AWARDS_1315:
        keyword_list = parse_award(award)

        for idx, tweet in enumerate(data):
          if idx in used_tweet_indices:
            continue

          if tweet_match(tweet, keyword_list):
            data_classified[award] = data_classified[award] + [tweet]
            used_tweet_indices.add(idx)

    res = {}

    for award, tweets in data_classified.items():
        res[award] = {}

        for tweet in tweets:
            curr_nominees = nominees[award]

            for nominee in curr_nominees:
                if nominee in tweet.lower():
                    if nominee not in res[award]:
                        res[award][nominee] = 0

                    res[award][nominee] = res[award][nominee] + 1

    res_object = json.dumps(res)

    with open("./data/INITIAL_RESULTS_2013.json", "w") as f:
        f.write(res_object)

if __name__ == "__main__":
    main()

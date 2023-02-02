import json
import re
import spacy
import time
import multiprocessing as mp

# MISC: See here (https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed) for SSL certificate issues w/ NLTK downloads
from nltk import pos_tag, word_tokenize, RegexpParser
from nltk.corpus import stopwords

NLP = spacy.load("en_core_web_sm")
PERSON_GRAMMAR = "NP: {<NNP><NNP>}"
STOP_WORDS = set(stopwords.words("english"))

def main():
    frame = AwardFrame()

    frame.load_tweets()

    # frame.generate_hosts()

    frame.generate_awards(False)
    frame.parse_award_keywords()

    #frame.type_system_nominee()
    frame.generate_nominees(False)

    # frame.generate_winners()

    # frame.print_results()

class AwardFrame:
    def __init__(self):
        self.results = {}
        self.nominee_type_system = {}

        self.award_groups = {}
        self.award_suffixes = ["motion picture", "comedy or musical", "television", "drama", "film"]
        self.award_keywords = {}

        self.winner_regex = [f"best(.*){suffix}" for suffix in self.award_suffixes]
        self.nominee_regex = ["(.*)nominee(.*)", "(.*)nominate(.*)"]
        self.hosts_regex = ["(.*)"]
        # self.nominee_regex = ["nominees for(.*) are(.*)", "(.*)is nominated for(.*)", "(.*)is[ ]?(?:a)?[ ]?nominee for(.*)", "(.*)are[ ]?(?:the)?[ ]?nominees for(.*)"]
        self.awards_regex = ["(.*)goes to(.*)", "(.*)wins(.*)", "(.*)takes home(.*)", "(.*)receives(.*)", "(.*)is the winner of(.*)"]
        self.tweets = []

    def load_tweets(self):
        with open("./data/gg2013.json") as f:
            tweets = json.load(f)

        self.tweets = list(map(lambda x: x["text"], tweets))

    def generate_awards(self, autofill = True):
        if not self.tweets:
            self.load_tweets()

        if autofill:
            with open("./data/nominees2013.json") as f:
                data = json.load(f)

            awards = data.keys()

            for award in awards:
                self.results[award] = {}
        else:
            results = {}

            for regex in self.winner_regex:
                for tweet in self.tweets:
                    match = re.search(regex, tweet)

                    if match:
                        award_name = match[0].lower().strip()
                        award_name_parsed = self.alpha_only_string(award_name)
                        award_name_parsed = " ".join(list(filter(lambda word: word not in STOP_WORDS or word in ["or", "in", "a"], award_name_parsed.split(" "))))

                        if award_name_parsed.startswith("best ") and any(award_name_parsed.endswith(suffix) for suffix in self.award_suffixes):
                          results[award_name_parsed] = results[award_name_parsed] + 1 if award_name_parsed in results else 1

            results = {k : v for k, v in results.items() if v > 5}

            awards = self.aggregate_awards(list(results.keys()))
            awards = self.parse_award_names(awards)

            self.results = {award : {} for award in awards}

    def aggregate_awards(self, awards):
      n = len(awards)

      res = []
      visited = [False for _ in range(n)]

      for i in range(n):
        if visited[i]:
          continue

        curr_award = awards[i]
        curr_award_keywords = " ".join(list(filter(lambda word: word not in STOP_WORDS, curr_award.split(" "))))
        matches = [curr_award]

        for j in range(n):
          if visited[j] or i == j:
            continue

          other_award = awards[j]
          other_award_keywords = " ".join(list(filter(lambda word: word not in STOP_WORDS, other_award.split(" "))))

          if curr_award_keywords == other_award_keywords: # TODO: Maybe make the aggregation method here slightly less strict
            matches.append(other_award)
            visited[j] = True

        if len(matches) > 1:
          max_index = matches.index(max(matches, key = len))

          longest_award = matches[max_index]
          res.append(longest_award)

          matches.pop(max_index)
          self.award_groups[longest_award] = matches
        else:
          res.append(curr_award)

        visited[i] = True

      return res

    def parse_award_names(self, awards):
        res = []

        for award in awards:
          pos_tags = pos_tag(word_tokenize(award))

          for suffix in self.award_suffixes:
            suffix_loc = award.find(suffix)

            if suffix_loc != -1:
              last_non_suffix_word = award[:suffix_loc - 1].split(" ")[-1]
              pos_tag_last_word = list(filter(lambda x: x[0] == last_non_suffix_word, pos_tags))[0][1]

              if pos_tag_last_word != "NN" or len(pos_tags) < 5:
                res.append(award)
              else:
                new_award_name = award[:suffix_loc - 1] + " - " + award[suffix_loc:]

                if award in self.award_groups:
                  similar_names = self.award_groups[award]
                  self.award_groups[new_award_name] = similar_names

                  del self.award_groups[award]

                res.append(new_award_name)

              break

        return res

    def parse_award_keywords(self):
      if not self.results:
        self.generate_awards()

      for award in self.results.keys():
        if award in self.award_groups:
          award_group = [award] + self.award_groups[award]

          for award_candidate in award_group:
            self.award_keywords[award_candidate] = list(filter(lambda word: word not in STOP_WORDS and word != "-", award.split(" ")))
        else:
          self.award_keywords[award] = list(filter(lambda word: word not in STOP_WORDS and word != "-", award.split(" ")))

    def generate_nominees(self, autofill = True):
        if not self.tweets:
            self.load_tweets()

        if autofill:
            with open("./data/nominees2013.json") as f:
                data = json.load(f)

            for award, nominee_list in data.items():
                self.results[award]["nominees"] = nominee_list
        else:
            awards = self.results.keys()

            # Parallelizing reference: https://www.machinelearningplus.com/python/parallel-processing-python/
            pool = mp.Pool(mp.cpu_count())
            results = pool.map(self.get_nominee_from_award, [award for award in awards])
            pool.close()

            for result in results:
              award = result[0]
              result.pop(0)

              self.results[award]["nominees"] = result

        self.write_to_file(self.results, "results_award_nominees.json")
        self.print_results()

    def get_nominee_from_award(self, award):
      results = {}
      award_type = ""

      if "actor" in award or "actress" in award:
        award_type = "person"
      else:
        award_type = "film"

      award_group = [award] + self.award_groups[award] if award in self.award_groups else [award]
      related_tweets = []

      for tweet in self.tweets:
        tweet_list = self.alpha_only_string(tweet).split(" ")

        for award_candidate in award_group:
          all_keywords = self.award_keywords[award_candidate]
          num_keywords = 0

          for keyword in all_keywords:
            if keyword in tweet_list:
              num_keywords += 1

          if num_keywords / len(all_keywords) >= 0.5:
              related_tweets.append(tweet)
              break

      for tweet in related_tweets:
        tweet = self.alpha_only_string(tweet)
        doc = NLP(tweet)

        if award_type == "person":
          nltk_nominees = self.match_grammar_from_tweet(tweet)
          spacy_nominees = []

          for word in doc.ents:
            if word.label_ == "PERSON":
              spacy_nominees.append(str(word))

          nominee_candidates = list(set(nltk_nominees) & set(spacy_nominees))

          for nominee_candidate in nominee_candidates:
            if nominee_candidate in results:
              results[nominee_candidate] = results[nominee_candidate] + 1
            else:
              results[nominee_candidate] = 1
        else:
          doc = NLP(tweet)
          for chunk in doc.noun_chunks:
            nominee_candidate = chunk.text

            if nominee_candidate in results:
              results[nominee_candidate] = results[nominee_candidate] + 1
            else:
              results[nominee_candidate] = 1

      top_results = sorted(results.items(), key = lambda x: x[1], reverse = True)[:10]
      top_results = [award] + [result[0] for result in top_results]
      return top_results

    def match_grammar_from_tweet(self, tweet):
      tweet = word_tokenize(tweet)
      tweet = pos_tag(tweet)
      chunk = RegexpParser(PERSON_GRAMMAR)
      chunk_tree = chunk.parse(tweet)

      matches = list(chunk_tree.subtrees(filter = lambda tree: tree.label() == "NP"))
      return list(map(lambda x : x[0][0] + " " + x[1][0], matches))

    def type_system_nominee(self):
        pass

    def generate_winners(self):
        if not self.tweets:
            self.load_tweets()

        results = {award : {} for award in self.results.keys()}

        for award in self.results.keys():
            subres = {nominee : 0 for nominee in self.results[award]["nominees"]}
            results[award] = subres

        # TODO: Instead of regex, maybe try nltk, e.g. named entity?
        # Check for noun phrases, see if one includes the word "award", other one is a nominee hopefully(?)
        for index, regex in enumerate(self.awards_regex):
            for tweet in self.tweets:
                match = re.search(regex, tweet)

                if match:
                    x, y = "", ""
                    if index == 0:
                        y, x = match.group(1), match.group(2)
                    else:
                        x, y = match.group(1), match.group(2)

                    curr_award, curr_winner = None, None

                    for award in self.results.keys():
                        if award in y.lower():
                            curr_award = award
                            break

                    if curr_award:
                        for nominee in self.results[curr_award]["nominees"]:
                            if nominee in x.lower():
                                curr_winner = nominee
                                break

                    if curr_winner:
                        results[curr_award][curr_winner] = results[curr_award][curr_winner] + 1

            for award in self.results.keys():
                max_votes = -1
                winner = None

                for nominee, votes in results[award].items():
                    if votes > max_votes:
                        max_votes = votes
                        winner = nominee

                self.results[award]["winner"] = winner

    def generate_presenters(self):
      pass

    def generate_hosts(self):
      if not self.tweets:
        self.load_tweets()

      results = {}

      for tweet in self.tweets:
        if "host" in tweet.lower() and "next year" not in tweet.lower():
          doc = NLP(tweet)
          for ent in doc.ents:
            if ent.label_ == 'PERSON' and len(ent.text.split(" ")) > 1:
              if ent.text in results:
                results[ent.text] += 1
              else:
                results[ent.text] = 1

      print(sorted(results.items(), key=(lambda x: x[1]), reverse=True)[0:2])

    def alpha_only_string(self, s):
      # TODO: Get rid of @[...]
      s = re.sub("[^a-zA-Z ]", "", s)
      s = re.sub(' +', ' ', s)
      return s

    def write_to_file(self, data, filename):
      json_obj = json.dumps(data)

      with open(f"./data/{filename}", "w") as f:
        f.write(json_obj)

    def print_winners(self):
        for award in self.results.keys():
            print(award, ":", self.results[award]["winner"])

    def print_results(self):
        print(self.results)


if __name__ == "__main__":
    main()

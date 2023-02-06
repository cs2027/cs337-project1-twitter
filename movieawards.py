import json
import re
import spacy
import multiprocessing as mp
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

# MISC: See here (https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed) for SSL certificate issues w/ NLTK downloads
from nltk import pos_tag, word_tokenize, RegexpParser
from nltk.corpus import stopwords

cloud_config= {
  'secure_connect_bundle': 'secure-connect-nlp.zip'
}

NLP = spacy.load("en_core_web_sm")
PERSON_GRAMMAR = "NP: {<NNP><NNP>}"
STOP_WORDS = set(stopwords.words("english"))

def main():
    frame = AwardFrame()

    frame.load_tweets()

    frame.generate_awards(False)
    frame.parse_award_keywords()

    frame.generate_related_tweets()

    frame.generate_presenters()

    frame.generate_nominees(False)

    frame.generate_winners()

    frame.generate_hosts()

    frame.print_results()
    frame.write_to_file(frame.results, "RESULTS_2013.json")

class AwardFrame:
    def __init__(self):
        self.results = {}

        self.award_groups = {}
        self.award_suffixes = ["motion picture", "comedy or musical", "television", "drama", "film"]
        self.award_keywords = {}

        self.related_tweets_nominees = {}
        self.related_tweets_presenters = {}

        self.winner_regex = [f"best(.*){suffix}" for suffix in self.award_suffixes]
        self.nominee_regex = ["(.*)nominee(.*)", "(.*)nominate(.*)"]
        self.hosts_regex = ["(.*)"]
        self.awards_regex = ["(.*)goes to(.*)", "(.*)wins(.*)", "(.*)takes home(.*)", "(.*)receives(.*)", "(.*)is the winner of(.*)"]

        self.names_cache = {}
        self.titles_cache = {}

        self.tweets = []

    @classmethod
    def _setup(cls):
      auth_provider = PlainTextAuthProvider('GPteigKOPCAybnALbhbZsFUd', 'uBjcp8pcknnG+Ja3tkRX+5I-Ng9+3KLt0p-BC1LDCOTBZLftZ-AkIZNMbxJMXMj.kz08-cHmpv46UwlgSh4J8sK_FgW.bOdsrf3JQdQx6JT4NAHtYAN+FZ+3gvFFdYtp')
      cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
      cls.session = cluster.connect()

    @classmethod
    def lookup_name(cls, name):
      res = cls.session.execute(f"SELECT role FROM imdb.names WHERE name = '{name}'").one()

      if res:
        return res
      else:
        None

    @classmethod
    def lookup_title(cls, title):
      res = cls.session.execute(f"SELECT * FROM imdb.titles WHERE title = '{title}'").one()

      if res:
        return res
      else:
        None

    def load_tweets(self):
        with open("./data/gg2013.json") as f:
            tweets = json.load(f)

        self.tweets = list(map(lambda x: x["text"], tweets))

    def generate_related_tweets(self):
      print("STARTING RELATED TWEETS")

      if not self.tweets:
        self.load_tweets()

      awards = self.results.keys()

      for award in awards:
        self.related_tweets_nominees[award] = {}
        self.related_tweets_presenters[award] = {}

      pool = mp.Pool(mp.cpu_count(), initializer=self._setup, initargs=())
      results = pool.map(self.get_related_tweets_from_award, [award for award in awards])
      pool.close()

      results = []

      for result in results:
        [award, related_tweets_nominees, related_tweets_presenters] = result

        self.related_tweets_nominees[award] = related_tweets_nominees
        self.related_tweets_presenters[award] = related_tweets_presenters

      print("FINISHING RELATED TWEETS")

    def get_related_tweets_from_award(self, award):
      award_group = [award] + self.award_groups[award] if award in self.award_groups else [award]
      related_tweets_nominees = []
      related_tweets_presenters = []

      for tweet in self.tweets:
        tweet_list = self.alpha_only_string(tweet).split(" ")

        for award_candidate in award_group:
          all_keywords = self.award_keywords[award_candidate]
          num_keywords = 0

          for keyword in all_keywords:
            if keyword in tweet_list:
              num_keywords += 1

          if num_keywords / len(all_keywords) >= 0.5:
              related_tweets_nominees.append(tweet)

              if "present" in tweet.lower():
                related_tweets_presenters.append(tweet)

              break

        return [award, related_tweets_nominees, related_tweets_presenters]

    def generate_awards(self, autofill = True):
        print("STARTING GENERATE AWARDS")

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

        print("ENDING GENERATE AWARDS")

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
        print("STARTING GENERATE NOMINEES")

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
            pool = mp.Pool(mp.cpu_count(), initializer=self._setup, initargs=())
            results = pool.map(self.get_nominee_from_award, [award for award in awards])
            pool.close()

            results = []

            for award in awards:
              results.append(self.get_nominee_from_award(award))

            for result in results:
              award = result[0]
              result.pop(0)

              self.results[award]["nominees"] = result

        self.write_to_file(self.results, "results_award_nominees.json")
        self.print_results()

        print("ENDING GENERATE AWARDS")

    def get_presenter_from_award(self, award):
      results = {}

      for tweet in self.related_tweets_presenters[award]:
        tweet = self.alpha_only_string(tweet)
        doc = NLP(tweet)

        nltk_nominees = self.match_grammar_from_tweet(tweet)
        spacy_nominees = []

        for word in doc.ents:
          if word.label_ == "PERSON":
            nominee = str(word)
            spacy_nominees.append(nominee)

        nominee_candidates = list(set(nltk_nominees) & set(spacy_nominees))

        for nominee_candidate in nominee_candidates:
          if nominee_candidate.startswith("RT"):
            nominee_candidate = nominee_candidate.split(" ")

            if len(nominee_candidate) <= 2:
              continue

            nominee_candidate = " ".join(nominee_candidate[2:])

          if self.contains_stopword(nominee_candidate) or "goldenglobe" in nominee_candidate.replace(" ", "").lower():
            continue

          if nominee_candidate in results:
            results[nominee_candidate] = results[nominee_candidate] + 1
          else:
            results[nominee_candidate] = 1

      top_results = sorted(results.items(), key = lambda x: x[1], reverse = True)[:3]
      top_results = [award] + [result[0] for result in top_results]
      return top_results

    def get_nominee_from_award(self, award):
      results = {}
      award_type = ""

      if "actor" in award:
        award_type = "actor"
      elif "actress" in award:
        award_type = "actress"
      else:
        award_type = "film"

      for tweet in self.related_tweets_nominees[award]:
        tweet = self.alpha_only_string(tweet)
        doc = NLP(tweet)

        if award_type == "actor" or award_type == "actress":
          nltk_nominees = self.match_grammar_from_tweet(tweet)
          spacy_nominees = []

          for word in doc.ents:
            if word.label_ == "PERSON":
              nominee = str(word)
              spacy_nominees.append(nominee)

          nominee_candidates = list(set(nltk_nominees) & set(spacy_nominees))

          for nominee_candidate in nominee_candidates:
            if nominee_candidate.startswith("RT"):
              nominee_candidate = nominee_candidate.split(" ")

              if len(nominee_candidate) <= 2:
                continue

              nominee_candidate = " ".join(nominee_candidate[2:])

            if self.contains_stopword(nominee_candidate) or "goldenglobe" in nominee_candidate.replace(" ", "").lower():
              continue

            if nominee_candidate in results:
              results[nominee_candidate] = results[nominee_candidate] + 1
            else:
              results[nominee_candidate] = 1
        else:
          nominee_candidates = self.get_proper_nouns(tweet)

          for nominee_candidate in nominee_candidates:
            if nominee_candidate.startswith("RT"):
              nominee_candidate = nominee_candidate.split(" ")

              if len(nominee_candidate) <= 2:
                continue

              nominee_candidate = " ".join(nominee_candidate[2:])

            if self.contains_stopword(nominee_candidate) or "goldenglobe" in nominee_candidate.replace(" ", "").lower():
              continue

            nominee_lookup = self.lookup_title(nominee_candidate)

            if nominee_lookup is not None:
              if nominee_candidate in results:
                results[nominee_candidate] = results[nominee_candidate] + 1
              else:
                results[nominee_candidate] = 1

      sorted_results = sorted(results.items(), key = lambda x: x[1], reverse = True)
      sorted_results = [result[0] for result in sorted_results]

      top_results = [award]
      i = 0

      if award_type == "actor" or award_type == "actress":
        for result in sorted_results:
          nominee_role = self.lookup_name(result)

          if nominee_role == "actor" and award_type == "actor":
            top_results.append(result)
            i += 1

          if nominee_role == "actress" and award_type == "actress":
            top_results.append(result)
            i += 1

          if i == 5:
            break
      else:
        for result in sorted_results:
          nominee_role = self.lookup_title(result)

          if nominee_role:
            top_results.append(result)
            i += 1

          if i == 5:
            break

      return top_results

    def get_proper_nouns(self, tweet):
      doc = NLP(tweet)
      res = []

      curr_pos = None
      curr_phrase = None

      for word in doc:
        if word.pos_ == "PROPN":
          if curr_pos != "PROPN":
            curr_pos = "PROPN"
            curr_phrase = str(word)
          else:
            curr_phrase += " "
            curr_phrase += str(word)
        else:
          if curr_pos == "PROPN":
            curr_pos = None
            res.append(curr_phrase)

            curr_phrase = None

      if curr_pos == "PROPN" and curr_phrase is not None:
        res.append(curr_phrase)

      return res

    def match_grammar_from_tweet(self, tweet):
      tweet = word_tokenize(tweet)
      tweet = pos_tag(tweet)
      chunk = RegexpParser(PERSON_GRAMMAR)
      chunk_tree = chunk.parse(tweet)

      matches = list(chunk_tree.subtrees(filter = lambda tree: tree.label() == "NP"))
      return list(map(lambda x : x[0][0] + " " + x[1][0], matches))

    def generate_winners(self):
        print("STARTING GENERATE WINNERS")

        if not self.tweets:
            self.load_tweets()

        results = {award : {} for award in self.results.keys()}

        for award in self.results.keys():
            subres = {nominee : 0 for nominee in self.results[award]["nominees"]}
            results[award] = subres

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

            print("ENDING GENERATE WINNERS")

    def generate_presenters(self):
      print("STARTING GENERATE PRESENTERS")

      if not self.tweets:
        self.load_tweets()

      awards = self.results.keys()

      pool = mp.Pool(mp.cpu_count(), initializer=self._setup, initargs=())
      results = pool.map(self.get_presenter_from_award, [award for award in awards])
      pool.close()

      results = []

      for award in awards:
        results.append(self.get_presenter_from_award(award))

      for result in results:
        award = result[0]
        result.pop(0)

        self.results[award]["presenters"] = result

      self.write_to_file(self.results, "presenters.json")
      self.print_results()

      print("ENDING GENERATE NOMINEES")

    def generate_hosts(self):
      print("STARTING GENERATE HOSTS")

      if not self.tweets:
        self.load_tweets()

      results = {}

      for tweet in self.related_tweets_hosts():
        if "host" in tweet.lower() and "next year" not in tweet.lower():
          doc = NLP(tweet)
          for ent in doc.ents:
            if ent.label_ == 'PERSON' and len(ent.text.split(" ")) > 1:
              if ent.text in results:
                results[ent.text] += 1
              else:
                results[ent.text] = 1

      results = sorted(results.items(), key=(lambda x: x[1]), reverse=True)[0:2]
      hosts = [x[0] for x in results]
      self.results["hosts"] = hosts

      print("ENDING GENERATE HOSTS")

    def alpha_only_string(self, s):
      s = re.sub("[^a-zA-Z ]", "", s)
      s = re.sub(' +', ' ', s)
      return s

    def contains_stopword(self, s):
      stopwords = list(filter(lambda word: word in STOP_WORDS, s.split(" ")))
      return len(stopwords) > 0

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

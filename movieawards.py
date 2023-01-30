import json
import re

# MISC: See here (https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed) for SSL certificate issues w/ NLTK downloads
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

def main():
    frame = AwardFrame()

    frame.load_tweets()

    frame.generate_awards()
    frame.parse_award_keywords()

    frame.type_system_nominee()
    frame.generate_nominees(False)

    # frame.generate_winners()

    # frame.print_results()

class AwardFrame:
    def __init__(self):
        self.results = {}
        self.nominee_type_system = {}

        self.award_suffixes = ["motion picture", "comedy or musical", "television", "drama", "film"]
        self.award_keywords = {}

        self.winner_regex = [f"best(.*){suffix}" for suffix in self.award_suffixes]
        self.nominee_regex = ["nominees for(.*) are(.*)", "(.*)is nominated for(.*)", "(.*)is[ ]?(?:a)?[ ]?nominee for(.*)", "(.*)are[ ]?(?:the)?[ ]?nominees for(.*)"]
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

          if curr_award_keywords == other_award_keywords:
            matches.append(other_award)
            visited[j] = True

        if len(matches) > 1:
          longest_award = max(matches, key = len)
          res.append(longest_award)
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
                res.append(award[:suffix_loc - 1] + " - " + award[suffix_loc:])

              break

        return res

    def parse_award_keywords(self):
      if not self.results:
        self.generate_awards()

      for award in self.results.keys():
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
            # TODO: Regex + type checking system
            results = {award : {} for award in self.results.keys()}

            for index, regex in enumerate(self.awards_regex):
                for tweet in self.tweets:
                    match = re.search(regex, tweet.lower())

                    if match:
                        x, y = "", ""
                        if index == 0:
                            y, x = match.group(1), match.group(2)
                        else:
                            x, y = match.group(1), match.group(2)

                        curr_award = None

                        for award in self.results.keys():
                            y_list = self.alpha_only_string(y.lower()).split(" ")
                            all_keywords = self.award_keywords[award]

                            num_keywords = 0

                            for keyword in all_keywords:
                              if keyword in y_list:
                                num_keywords += 1

                            if num_keywords / len(all_keywords) >= 0.5:
                                curr_award = award
                                break

                        if curr_award:
                            # NOTE: Currently ignoring possible lists of nominees
                            # if index == 0 or index == 3:
                            #     if "," in x:
                            #         nominees = [s.strip() for s in x.split(",")]
                            #         for nominee in nominees:
                            #             results[curr_award][nominee] = results[curr_award][nominee] + 1 if nominee in results[curr_award] else 1
                            # else:
                            if x.startswith("rt @"):
                              x = x[4:]

                            x = self.alpha_only_string(x.lower().strip())
                            results[curr_award][x] = results[curr_award][x] + 1 if x in results[curr_award] else 1

            for award in results:
                votes = sorted(results[award], key = lambda x: x[1], reverse = True)
                self.results[award]["nominees"] = votes

        self.print_results()

    def type_system_nominee(self):
        pass

    def generate_winners(self):
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

    def alpha_only_string(self, s):
      s = re.sub("[^a-zA-Z ]", "", s)
      s = re.sub(' +', ' ', s)
      return s

    def print_winners(self):
        for award in self.results.keys():
            print(award, ":", self.results[award]["winner"])

    def print_results(self):
        print(self.results)


if __name__ == "__main__":
    main()

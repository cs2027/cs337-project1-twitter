import json
import re

def main():
    frame = AwardFrame()

    frame.generate_awards()

    frame.type_system_nominee()
    frame.generate_nominees(False)

    #frame.generate_winners()

    #print(frame.results)

class AwardFrame:
    def __init__(self):
        self.results = {}
        self.nominee_type_system = {}
        self.nominee_regex = ["nominees for(.*) are(.*)", "(.*)is nominated for(.*)", "(.*)is[ ]?(?:a)?[ ]?nominee for(.*)", "(.*)are[ ]?(?:the)?[ ]?nominees for(.*)"]
        self.winner_regex = []
        self.awards_regex = ["(.*)goes to(.*)", "(.*)wins(.*)", "(.*)takes home(.*)", "(.*)receives(.*)", "(.*)is the winner of(.*)"]

    def generate_awards(self, autofill = True):
        if autofill:
            with open("./data/nominees2013.json") as f:
                data = json.load(f)

            awards = data.keys()

            for award in awards:
                self.results[award] = {}
        else:
            # TODO: Regex
            pass

    def generate_nominees(self, autofill = True):
        if autofill:
            with open("./data/nominees2013.json") as f:
                data = json.load(f)

            for award, nominee_list in data.items():
                self.results[award]["nominees"] = nominee_list
        else:
            # TODO: Regex + type checking system
            results = {award : {} for award in self.results.keys()}
            with open("./data/gg2013.json") as f:
                tweets = json.load(f)

            tweets = list(map(lambda x: x["text"], tweets))

            for index, regex in enumerate(self.awards_regex):
                for tweet in tweets:
                    match = re.search(regex, tweet)

                    if match:
                        x, y = "", ""
                        if index == 0:
                            y, x = match.group(1), match.group(2)
                        else:
                            x, y = match.group(1), match.group(2)

                        
                        curr_award = None

                        for award in self.results.keys():
                            if award in y.lower():
                                curr_award = award
                                break
                        if curr_award:
                            if index == 0 or index == 3:
                                if "," in x:
                                    nominees = [s.strip() for s in x.split(",")]
                                    for nominee in nominees:
                                        if nominee in results[curr_award]:
                                            results[curr_award][nominee] = results[curr_award][nominee] + 1
                                        else:
                                            results[curr_award][nominee] = 1 
                            else:
                                x = x.lower().strip()
                                if x in results[curr_award]:
                                    results[curr_award][x] = results[curr_award][x] + 1
                                else:
                                    results[curr_award][x] = 1 
           
            for award in results:
                votes = sorted(results[award], key = lambda x: x[1], reverse = True)
                self.results[award]["nominees"] = votes[:5]

          




    def type_system_nominee(self):
        pass

    def generate_winners(self):
        with open("./data/gg2013.json") as f:
            tweets = json.load(f)

        tweets = list(map(lambda x: x["text"], tweets))

        results = {award : {} for award in self.results.keys()}

        for award in self.results.keys():
            subres = {nominee : 0 for nominee in self.results[award]["nominees"]}
            results[award] = subres

        for index, regex in enumerate(self.awards_regex):
            for tweet in tweets:
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

    def print_winners(self):
        for award in self.results.keys():
            print(award, ":", self.results[award]["winner"])

    def print_results(self):
        print(self.results)        

if __name__ == "__main__":
    main()

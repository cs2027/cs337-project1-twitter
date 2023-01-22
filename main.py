import json

OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
IDX = 3

def main():
    with open("./data/nominees2013.json", "r") as f:
        nominees = json.load(f)

    with open("./data/gg2013.json") as f:
        data = json.load(f)

    data = list(map(lambda x: x["text"], data))
    data_classified = {}

    for award in OFFICIAL_AWARDS_1315:
        for tweet in data:
            if award in tweet.lower():
                if award not in data_classified:
                    data_classified[award] = []

                data_classified[award] = data_classified[award] + [tweet]

    res = {}

    for award, tweets in data_classified.items():
        for tweet in tweets:
            curr_nominees = nominees[award]

            for nominee in curr_nominees:
                if nominee in tweet.lower():
                    if award not in res:
                        res[award] = {}

                    if nominee not in res[award]:
                        res[award][nominee] = 0

                    res[award][nominee] = res[award][nominee] + 1

    res_object = json.dumps(res)

    print(res_object)

    with open("./data/INITIAL_RESULTS_2013.json", "w") as f:
        f.write(res_object)

if __name__ == "__main__":
    main()
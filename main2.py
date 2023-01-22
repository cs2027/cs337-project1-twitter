import json

OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
IDX = 3

def main():


    with open("./data/gg2013.json") as f:
        tweets = json.load(f)

    tweets = list(map(lambda x: x["text"], tweets))

    with open("./data/nominees2013.json", "r") as f:
        nominees = json.load(f)
        allnominees = []
        for k, v in nominees.items():
            allnominees += v
    
    results = {}
    for nominee in allnominees:
        results[nominee] = 0
        for tweet in tweets:
            if nominee in tweet.lower():
                results[nominee] = results[nominee] + 1

    winners = {}
    for k, v in nominees.items():
        maxvotes = -1
        winner = ""
        for nominee in v:
            if results[nominee] > maxvotes:
                maxvotes = results[nominee]
                winner = nominee

        winners[k] = winner
    print(winners)

    res_object = json.dumps(winners)

    with open("./data/INITIAL_RESULTS2_2013.json", "w") as f:
        f.write(res_object)

    

if __name__ == "__main__":
    main()
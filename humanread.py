import json

with open("./data/RESULTS_2013.json", "r") as f:
    award_data = json.load(f)

with open("./data/RESULT_2013_HR.json", "w") as f:
    for award, data in award_data.items():
        if award != "hosts":
            f.write("Award: " + award + "\n")
            f.write("Presenters: " + ", ".join(data["presenters"]) + "\n")
            f.write("Nominess: " + ", ".join(data["nominees"]) + "\n") 
            f.write("Winner: " + data["winner"] + "\n")
            f.write("\n")
        else:
            f.write("Hosts: " + ", ".join(data) + "\n")
    
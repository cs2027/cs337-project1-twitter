import json
import sys

if len(sys.argv) < 2:
  exit(1)

year = int(sys.argv[1])

with open(f"./data/RESULTS_{year}.json", "r") as f:
    award_data = json.load(f)

with open(f"./data/RESULTS_{year}_HR.json", "w") as f:
    for award, data in award_data.items():
        if award != "hosts":
            f.write("Award: " + award + "\n")
            f.write("Presenters: " + ", ".join(data["presenters"]) + "\n")
            f.write("Nominees: " + ", ".join(data["nominees"]) + "\n")
            f.write("Winner: " + data["winner"] + "\n")
            f.write("\n")
        else:
            f.write("Hosts: " + ", ".join(data) + "\n")

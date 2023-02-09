To see the output in the two desired formats (JSON, human-readable), first run the "movieawards.py" file with
the desired year in the terminal (e.g. movieawards.py 2013) to see the results for the 2013 Golden Globes Show (/data/RESULTS_2013.json).

Then, to view the human-readable format run the "humanread.py" with the same year in the terminal
(e.g. humanread.py 2013), which takes the previous JSON file & converts it to a readable .json file format (/data/RESULTS_2013_HR.json).

For sentiment analysis, we ran the analysis on the hosts, golden globe parties, and winners. For sentiment analysis, run "sent_analysis.py" with a year as the command line argument. The results for winners will be in "sentanalysiswinners{year}.txt". 
The results for hosts are in "sentanalysishosts{year}.txt". The results include the overall sentiment (Positive or Negative) for winners and hosts including their sentiment scores. (The overall sentiment is the sum of sentiment polarities across all tweets.)
Note: Even though we don't have the 2015 data (gg2015.json), you can run sent_analysis with the year argument 2015 if you include the gg2015.json file.  
 The raw results of the analysis are in the /data/sentresults* files, the stats on the analysis (mean, max polarity, min polarity, etc.) are in /data/sentstats* files, and the final average sentiment of the hosts, golden globe parties, and winners are in the /data/finalanalysis* files.

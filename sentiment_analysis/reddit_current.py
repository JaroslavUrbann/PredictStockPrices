from psaw import PushshiftAPI
import csv

# Get reddit comments upto a date and store them into csv file
api = PushshiftAPI()

comments = api.search_comments(subreddit='teslamotors',     # input subreddit
                               after=1533815231,            # input date that is 2-3 days before today
                               filter=["body", "score"])

with open("TeslaComments_current.csv", "w", newline="") as f:   # change file name (it creates by itself)
    thewriter = csv.writer(f)
    for comment in comments:
        thewriter.writerow([comment.created_utc, comment.body, comment.score])

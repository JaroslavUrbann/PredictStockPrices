from psaw import PushshiftAPI
import csv

# Get all reddit comments and store them into csv file
api = PushshiftAPI()

comments = api.search_comments(subreddit='teslamotors',         # input subreddit
                               filter=["body", "score"])

with open("TeslaComments_history.csv", "w", newline="") as f:   # change file name (it creates by itself)
    thewriter = csv.writer(f)
    for comment in comments:
        thewriter.writerow([comment.created_utc, comment.body, comment.score])

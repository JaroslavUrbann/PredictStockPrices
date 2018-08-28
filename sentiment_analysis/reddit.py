from textblob import TextBlob
import praw

# Just some fun stuff
reddit = praw.Reddit(client_id="1XoNmFqnXgDABA",
                     client_secret="p-jgJ138aDH4rlM6StctW9g6H2o",
                     user_agent="subsentiment")

subreddits = ["teslamotors",
              "RealTesla",
              "google",
              "apple",
              "investing",
              "StockMarket",
              "Economics",
              "finance",
              "stocks"]


for subreddit in subreddits:
    sentiment = 0
    comments = list(reddit.subreddit(subreddit).comments(limit=None))
    n_comments = len(comments)
    for comment in comments:
        if comment.ups >= comment.downs:
            blob = TextBlob(comment.body)
            sentiment += blob.sentiment.polarity
            if comment.ups - comment.downs > 0:
                sentiment += blob.sentiment.polarity * (comment.ups - comment.downs)
                n_comments += comment.ups - comment.downs
    print(subreddit)
    print(sentiment/n_comments*100)
    print("      ", n_comments)

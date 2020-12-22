import csv
import praw


for j in [5, 10, 15, 20, 25, 30]:
    with open('lib_conserv' + str(j) + '.csv', mode='w', newline="", encoding="utf-8") as subreddit_data:
        fields = ['text', 'target']
        subreddit_writer = csv.DictWriter(subreddit_data, fieldnames=fields)
        subreddit_writer.writeheader()

        #############
        cl_id = 'SzqZtUpn4LG9zg'
        cl_scrt = 'obglC7k8apCcO-TjAJPwkXZdzkE'
        user = 'MLWebScraping'

        reddit = praw.Reddit(
            client_id=cl_id, client_secret=cl_scrt, user_agent=user)

        top_dem_posts = reddit.subreddit('Liberal').top(limit=j)
        for post in top_dem_posts:
            if len(post.comments) > 21:
                for i in range(1, 20):
                    subreddit_writer.writerow(
                        {'text': post.comments[i].body, 'target': 0})
            else:
                for i in range(1, len(post.comments)):
                    subreddit_writer.writerow(
                        {'text': post.comments[i].body, 'target': 0})

        top_rep_posts = reddit.subreddit('Conservative').top(limit=j)
        for post in top_rep_posts:
            if len(post.comments) > 21:
                for i in range(1, 20):
                    subreddit_writer.writerow(
                        {'text': post.comments[i].body, 'target': 1})
            else:
                for i in range(1, len(post.comments)):
                    subreddit_writer.writerow(
                        {'text': post.comments[i].body, 'target': 1})

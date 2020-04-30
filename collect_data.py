import tweepy
import csv
import pandas as pd

consumer_key = 'XXX'
consumer_secret = 'XXX'
access_token = 'XXX'
access_token_secret = 'XXX'



# get tweets from user's timeline
def get_all_tweets(screen_name):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    # use tweepy's api user_timeline to collect user's tweet
    all_the_tweets = []
    # use new_tweets to collect 200 tweets each time
    print('... downloading tweets.')
    new_tweets = api.user_timeline(screen_name=screen_name, count=201, tweet_mode="extended")
    # store all the tweets
    all_the_tweets.extend(new_tweets)
    oldest_tweet = all_the_tweets[-1].id - 1
    t_no = 201
    # collect more tweets till there is no tweet left
    while len(all_the_tweets) != t_no:
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest_tweet, tweet_mode="extended")
        t_no = len(all_the_tweets)
        all_the_tweets.extend(new_tweets)
        oldest_tweet = all_the_tweets[-1].id - 1

    tweets_without_retweet = []
    for tweet in all_the_tweets:
        # Check if it is a retweet. If yes, add the original tweet
        if hasattr(tweet, 'retweeted_status'):
            pass
        else:
            tweets_without_retweet.append(tweet)
    # # transforming the tweets into a 2D array that will be used to process
    # raw_tweets = [[tweet.id_str, tweet.created_at, tweet.full_text] for tweet in tweets_without_retweet]
    print('...%s tweets have been downloaded.' % len(all_the_tweets))

    all_df = tweet2df(tweets_without_retweet)
    output_path = '/Users/nanarua/Desktop/FYP/data/'
    output_path = output_path + screen_name + '_tweets.csv'
    # # transform the tweepy tweets into a 2D array that will populate the csv
    all_df.to_csv(output_path, encoding='utf-8', index=False)
    print('...totally %s tweets without tweets have been saved' % len(tweets_without_retweet))
    return all_df

    # # writing to the csv file
    # with open('/Users/nanarua/Desktop/FYP/data/' + screen_name + '_tweets.csv', 'w', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(raw_tweets)
    # return raw_tweets

# get tweets from my own data set(csv file)
def get_dataset(screen_name):
    raw_tweets = []
    with open('/Users/nanarua/Desktop/FYP/data/' + screen_name +'_tweets.csv')as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            raw_tweets.append(row)
    return raw_tweets

# get tweets from a larger data set (txt file)
def get_txt_dataset():
    raw_tweets = []
    file = open('/Users/nanarua/Desktop/FYP/data/Tweet.txt', 'r')
    raw_tweets = file.readlines()
    file.close()
    return raw_tweets

def tweet2df(tweets):
    columns = ['id', 'source', 'text', 'favorite_count', 'retweet_count', 'created_y', 'created_m', 'created_d',
               'created_h', 'created_min']
    data = [
        [tweet.id, tweet.source, tweet.full_text, tweet.favorite_count, tweet.retweet_count,
         tweet.created_at.year, tweet.created_at.month, tweet.created_at.day, tweet.created_at.hour,
         tweet.created_at.minute]
        for tweet in tweets]
    df = pd.DataFrame(data, columns=columns)
    return df




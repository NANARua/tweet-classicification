import collect_data
import processing
import csv
import numpy
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from skfuzzy.cluster import cmeans
import skfuzzy


# def kmeans_clustering(tweet_list, n):
#     km_cluster = KMeans(n_clusters=n, max_iter=200, n_init=20, \
#                         init='k-means++', n_jobs=1)
#     km_cluster.fit(tfidf_matrix)
#     labels = km_cluster.labels_
#     clustering_result = zip(tweets_list, labels)
#     return clustering_result

# about 80% accuracy
def ward_clustering(tweet_list, n):
    ward = AgglomerativeClustering(n_clusters=n, linkage='ward')
    ward.fit(tfidf)
    clustering_result = zip(tweets_list, ward.labels_)
    return clustering_result

# BAD result even after adjust parameter
def AP_clustering(tweet_list):
    af = AffinityPropagation(damping = 0.85, preference = -200).fit(tfidf)
    clustering_result = zip(tweets_list, af.labels_)
    return clustering_result





# raw_tweets = collect_data.get_dataset()
# tweets_list = processing.tweets_processing(raw_tweets)
raw_tweets = collect_data.get_txt_dataset()
tweets_list = processing.tweets_txtprocessing(raw_tweets)
tfidf = processing.get_tfidf(tweets_list)
tfidf = processing.decomposition(tfidf, 3)
result = ward_clustering(tweets_list, 89)
with open('result.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['tweet', 'groups'])
    writer.writerows(result)

import collect_data
import processing
import pandas as pd

if __name__ == '__main__':
    raw_tweets = collect_data.get_dataset()
    tweets = processing.tweets_processing(raw_tweets)
    sentence_vectors = processing.bow(tweets)
    tfidf = processing.get_tfidf(tweets)
    tfidf = processing.decomposition(tfidf, 3)
    matrix, fpc = processing.fuzzyc(tfidf, 8, 2)

    # test. find the appropriate parameter
    # fpc_result = []
    # temp = []
    # for i in range(2, 8):
    #     new_tfidf = processing.decomposition(tfidf, n=i)
    #     for n in range(15, 25):
    #         # print('n = ', n/10)
    #         matrix, fpc = processing.fuzzyc(new_tfidf, n_cluster=8, m=n/10)
    #         temp.append(fpc)
    #     fpc_result.append(temp)
    #     temp = []

    # test = pd.DataFrame(data=fpc_result)
    # test.to_csv('/Users/nanarua/Desktop/FYP/data/fpc_selection.csv', encoding='gbk')
    print('done')

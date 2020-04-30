from gensim.models import CoherenceModel
from collect_data import *
import processing
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
from nltk.stem.porter import PorterStemmer
import gensim
from gensim.models import HdpModel
import pandas as pd

def get_topic(texts, topics_num):
    # tokenize document string
    # p_stemmer = PorterStemmer()
    # texts = []
    # for tweet in tweets:
    #     tokens = RegexpTokenizer(r'\w+').tokenize(tweet)
    #     stemmed_tokens = [p_stemmer.stem(i) for i in tokens]
    #     texts.append(stemmed_tokens)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=topics_num,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    print(lda_model.print_topics(num_topics=topics_num, num_words=5))

    # coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    # coherence_values = coherence_model_lda.get_coherence()
    # print(lda_model)
    # print(coherence_lda)

    return lda_model


def compute_coherence_values(texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    # texts = []
    # p_stemmer = PorterStemmer()
    # for tweet in tweets:
    #     tokens = RegexpTokenizer(r'\w+').tokenize(tweet)
    #     stemmed_tokens = [p_stemmer.stem(i) for i in tokens]
    #     texts.append(stemmed_tokens)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=num_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
        coherencemodel = CoherenceModel(model=model, texts=texts,
                                        dictionary=dictionary,
                                        coherence='c_v')
        model_list.append(model)
        coherence_values.append(coherencemodel.get_coherence())
        # model = gensim.models.ldamodel.LdaModel(corpus=corpus,
        #                                         id2word=dictionary,
        #                                         num_topics=num_topics,
        #                                         random_state=100,
        #                                         update_every=1,
        #                                         chunksize=100,
        #                                         passes=10,
        #                                         alpha='auto',
        #                                         per_word_topics=True)        model_list.append(model)
        # coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        # coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


# get raw tweets
screen_name = 'gcosma1'
input_path = '/Users/nanarua/Desktop/FYP/data/' + screen_name +'_tweets.csv'
tweets_df = pd.read_csv(input_path)
tweets = []
for index, row in tweets_df.iterrows():
    tweets.append(row['text'])
texts = processing.tweets_processing(tweets)
texts = processing.tokenizer(texts)

# dictionary, texts, corpus, lda_model = get_topic(tweets_text, 9)
# hdp = HdpModel(corpus, dictionary)
# topic_info = hdp.print_topics(num_topics=8, num_words=5)
# print(topic_info)

# use coherence to determine the number of topics
limit = 21;
start = 5;
step = 1;
x = range(start, limit, step)
model_list, coherence_values = compute_coherence_values(texts=texts, start=start, limit=limit, step=step)
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
# according to the coherence value result, estimate the best topics number for clustering
best_topics_num = x[coherence_values.index(max(coherence_values))]
print('The best topics number is ', best_topics_num)
lda_model = get_topic(texts, best_topics_num)
print(lda_model)

# -*- coding: utf-8 -*-
import numpy as np
import re
from textblob import TextBlob
from textblob import Word
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import heapq
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import skfuzzy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# (csv file)raw_tweets are the data, N is column that contain data
def tweets_processing(raw_tweets):
    print("tweets processing...")
    tweets_list = []
    # read the data collected in get_all_tweets()
    lines = raw_tweets
    for line in lines:
        text = line
        # use clean_text() to remove noise elements
        text = clean_text(text)
        text = remove_number(text)
        text = normalization(text)
        text = remove_stopwords(text)
        tweets_list.append(text)
    return tweets_list

#
def tweets_txtprocessing(raw_tweets):
    print("tweets processing...")
    tweets_list = []
    # read the data collected in get_all_tweets()
    lines = raw_tweets
    for line in lines:
        text = line
        # use clean_text() to remove noise elements
        text = clean_text(text)
        text = remove_number(text)
        text = remove_stopwords(text)
        text = normalization(text)
        tweets_list.append(text)
    return tweets_list

def clean_text(text):
    # remove user-names and re-tweet marks
    text = re.sub(r'RT', '', text)
    text = re.sub(r'@[^\s]+', '', text)
    # remove hashtags
    #[^\s]+
    text = re.sub('#[^\s]+', '', text)
    # remove url
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)
    text = re.sub(r'http\S+', '', text)
    # remove other things
    text = re.sub(r'\n', '', text)
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)
    # check characters to see if they are in punctuation
    text = [char for char in text if char not in string.punctuation]
    # join the characters again to form the string
    text = ''.join(text)
    # convert text to lower case
    text = text.lower()
    return text


# def token_split(text):
#     terms = TextBlob(text).words.singularize()
#     result = []
#     for word in terms:
#         expected_str = Word(word)
#         expected_str = expected_str.lemmatize("v")
#         result.append(expected_str)
#     return result

def get_tfidf_matrix(tweets_list):
    tfidf_vectorizer = TfidfVectorizer(lowercase=True)
    tfidf_vectorizer.fit(tweets_list)
    tfidf_matrix = tfidf_vectorizer.transform(tweets_list)
    return tfidf_matrix


def remove_number(text):
    no_digits = []
    # Iterate through the string, adding non-numbers to the no_digits list
    for i in text:
        if not i.isdigit():
            no_digits.append(i)
    # Now join all elements of the list with '', puts all of the characters together.
    return ''.join(no_digits)


# use nltk library to remove stopwords
def remove_stopwords(text):
    text = word_tokenize(text)
    # add extra stopwords to the list
    extra_stopwords = ['w', 'v', 'via', 'j', 'r', 'amp']
    stop_words = set(stopwords.words('english'))
    final_stopwords = stop_words.union(extra_stopwords)
    filtered_sentence = []
    for w in text:
        if w not in final_stopwords:
            filtered_sentence.append(w)
    return " ".join(filtered_sentence)


# perform Text Normalization(lemmatization) using TextBlob
def normalization(text):
    lem = []
    for i in text.split():
        word1 = Word(i).lemmatize("n")
        word2 = Word(word1).lemmatize("v")
        word3 = Word(word2).lemmatize("a")
        lem.append(Word(word3).lemmatize())
    return " ".join(lem)


def bow(tweets):
    word_frequency = {}
    for tweet in tweets:
        tokens = nltk.word_tokenize(tweet)
        for token in tokens:
            if token not in word_frequency.keys():
                word_frequency[token] = 1
            else:
                word_frequency[token] += 1
    # filter down to the 200 most frequently occurring words
    most_freq = heapq.nlargest(200, word_frequency, key=word_frequency.get)
    # convert the tweets into their corresponding vector representation
    sentence_vectors = []
    for tweet in tweets:
        sentence_tokens = nltk.word_tokenize(tweet)
        sent_vec = []
        for token in most_freq:
            if token in sentence_tokens:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        sentence_vectors.append(sent_vec)
        # convert list of lists into matrix
    sentence_vectors = np.asarray(sentence_vectors)
    return sentence_vectors


def get_tfidf(corpus):
    print('calculating the tf-idf')
    # transform words into term frequency matrix
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    # calculate tf-idf based on term frequency matrix
    transformer = TfidfTransformer()
    print(transformer)
    tfidf = transformer.fit_transform(X)
    # Make the matrix sparse
    tfidf = tfidf * 10
    # transform list into array
    return tfidf.toarray()
    # return tfidf


def fuzzyc(tfidf, n_cluster, m):
    centre, u, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans(tfidf.T, c=n_cluster, m=m, error=0.00001, maxiter=1000000,
                                                          init=None)
    return u, fpc
    # return u


def decomposition(data, n):
    pca = PCA(n_components=n)  # loading PCA algorithm
    reduced = pca.fit_transform(data)
    return reduced

# def decomposition(data, n):
#     tsne = TSNE(n_components=n).fit_transform(data)
#     return tsne

def tokenizer(tweets_texts):
    texts =[]
    for text in tweets_texts:
        tokens = nltk.RegexpTokenizer(r'\w+').tokenize(text)
        texts.append(tokens)
    return texts

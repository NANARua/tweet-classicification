# -*- coding: utf-8 -*-
import projector
from sklearn.manifold import TSNE
from collections import Counter
from six.moves import cPickle
import gensim.models.word2vec as w2v
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing
import os
import sys
import io
import re
import json
from clean_text import  cleantext



# a few helpful function
# load the previously saved output from a function
def try_load_or_process(filename, processor_fn, function_arg):
    load_fn = None
    save_fn = None
    if filename.endswith("json"):
        load_fn = load_json
        save_fn = save_json
    else:
        load_fn = load_bin
        save_fn = save_bin
    if os.path.exists(filename):
        return load_fn(filename)
    else:
        ret = processor_fn(function_arg)
        save_fn(ret, filename)
        return ret

def print_progress(current, maximum):
    sys.stdout.write("\r")
    sys.stdout.flush()
    sys.stdout.write(str(current) + "/" + str(maximum))
    sys.stdout.flush()


def save_bin(item, filename):
    with open(filename, "wb") as f:
        cPickle.dump(item, f)

def load_bin(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return cPickle.load(f)

def save_json(variable, filename):
    with io.open(filename, "w", encoding="utf-8") as f:
        f.write(str(json.dumps(variable, indent=4, ensure_ascii=False)))

def load_json(filename):
    ret = None
    if os.path.exists(filename):
        try:
            with io.open(filename, "r", encoding="utf-8") as f:
                ret = json.load(f)
        except:
            pass
    return ret





def get_word_frequencies(corpus):
    frequencies = Counter()
    for sentence in corpus:
        for word in sentence:
            frequencies[word] += 1
    freq = frequencies.most_common()
    return freq

def clean_sentences(tokens):
    all_stopwords = load_json("stopwords-iso.json")
    extra_stopwords = []
    stopwords = None
    if all_stopwords is not None:
        stopwords = all_stopwords["fi"]
        stopwords += extra_stopwords
    ret = []
    max_s = len(tokens)
    for count, sentence in enumerate(tokens):
        if count % 50 == 0:
            print_progress(count, max_s)
        cleaned = []
        for token in sentence:
            if len(token) > 0:
                if stopwords is not None:
                    for s in stopwords:
                        if token == s:
                            token = None
                if token is not None:
                    if re.search("^[0-9\.\-\s\/]+$", token):
                            token = None
                if token is not None:
                    cleaned.append(token)
            if len(cleaned) > 0:
                ret.append(cleaned)
        return ret

def get_word2vec(sentences):
    min_frequency_val = 0
    num_workers = multiprocessing.cpu_count()
    num_features = 200
    epoch_count = 10
    sentence_count = len(sentences)
    w2v_file = os.path.join(save_dir, "word_vectors.w2v")
    word2vec = None
    if os.path.exists(w2v_file):
        print("w2v model loaded from " + w2v_file)
        word2vec = w2v.Word2Vec.load(w2v_file)
    else:
        word2vec = w2v.Word2Vec(sg=1,
                                seed=1,
                                workers=num_workers,
                                size=num_features,
                                min_count=min_frequency_val,
                                window=5,
                                sample=0)
    print("Building vocab...")
    word2vec.build_vocab(sentences)
    print("Word2Vec vocabulary length:", len(word2vec.wv.vocab))
    print("Training...")
    word2vec.train(sentences, total_examples=sentence_count, epochs=epoch_count)
    print("Saving model...")
    word2vec.save(w2v_file)
    return word2vec


def create_embeddings(word2vec):
    all_word_vectors_matrix = word2vec.wv.syn0
    num_words = len(all_word_vectors_matrix)
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    dim = word2vec.wv[vocab[0]].shape[0]
    embedding = np.empty((num_words, dim), dtype=np.float32)
    metadata = ""
    for i, word in enumerate(vocab):
        embedding[i] = word2vec.wv[word]
        metadata += word + "\n"
    metadata_file = os.path.join(save_dir, "metadata.tsv")
    with io.open(metadata_file, "w", encoding="utf-8") as f:
        f.write(metadata)

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    X = tf.Variable([0.0], name='embedding')
    place = tf.placeholder(tf.float32, shape=embedding.shape)
    set_x = tf.assign(X, place, validate_shape=False)
    sess.run(tf.global_variables_initializer())
    sess.run(set_x, feed_dict={place: embedding})

    summary_writer = tf.summary.FileWriter(save_dir, sess.graph)
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = 'embedding:0'
    embedding_conf.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(summary_writer, config)

    save_file = os.path.join(save_dir, "model.ckpt")
    print("Saving session...")
    saver = tf.train.Saver()
    saver.save(sess, save_file)

def calculate_t_sne():
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    arr = np.empty((0, dim0), dtype='f')
    labels = []
    vectors_file = os.path.join(save_dir, "vocab_vectors.npy")
    labels_file = os.path.join(save_dir, "labels.json")
    if os.path.exists(vectors_file) and os.path.exists(labels_file):
        print("Loading pre-saved vectors from disk")
        arr = load_bin(vectors_file)
        labels = load_json(labels_file)
    else:
        print("Creating an array of vectors for each word in the vocab")
        for count, word in enumerate(vocab):
            if count % 50 == 0:
                print_progress(count, vocab_len)
            w_vec = word2vec[word]
            labels.append(word)
            arr = np.append(arr, np.array([w_vec]), axis=0)
        save_bin(arr, vectors_file)
        save_json(labels, labels_file)

    x_coords = None
    y_coords = None
    x_c_filename = os.path.join(save_dir, "x_coords.npy")
    y_c_filename = os.path.join(save_dir, "y_coords.npy")
    if os.path.exists(x_c_filename) and os.path.exists(y_c_filename):
        print("Reading pre-calculated coords from disk")
        x_coords = load_bin(x_c_filename)
        y_coords = load_bin(y_c_filename)
    else:
        print("Computing T-SNE for array of length: " + str(len(arr)))
        tsne = TSNE(n_components=2, random_state=1, verbose=1)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(arr)
        x_coords = Y[:, 0]
        y_coords = Y[:, 1]
        print("Saving coords.")
        save_bin(x_coords, x_c_filename)
        save_bin(y_coords, y_c_filename)
    return x_coords, y_coords, labels, arr



if __name__ == '__main__':
    input_dir = "data"
    save_dir = "analysis"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("cleaning raw text")
    raw_input_file = os.path.join(input_dir, "gcosma1_tweets.csv")
    filename = os.path.join(save_dir, "data.json")
    processed = try_load_or_process(filename, cleantext, raw_input_file)
    print("Unique sentences: " + str(len(processed)))


    print("Getting word frequencies")
    filename = os.path.join(save_dir, "frequencies.json")
    frequencies = try_load_or_process(filename, get_word_frequencies, processed)
    vocab_size = len(frequencies)
    print("Unique words: " + str(vocab_size))
    print(frequencies)

    trimmed_vocab = []
    min_frequency_val = 6
    for item in frequencies:
        if item[1] >= min_frequency_val:
            trimmed_vocab.append(item[0])
    trimmed_vocab_size = len(trimmed_vocab)
    print("Trimmed vocab length: " + str(trimmed_vocab_size))
    filename = os.path.join(save_dir, "trimmed_vocab.json")
    save_json(trimmed_vocab, filename)

    print("Instantiating word2vec model")
    word2vec = get_word2vec(processed)
    vocab = word2vec.wv.vocab.keys()
    vocab_len = len(vocab)
    print("word2vec vocab contains " + str(vocab_len) + " items.")
    dim0 = word2vec.wv[list(vocab[0])].shape[0]
    print("word2vec items have " + str(dim0) + " features.")

    print("Creating tensorboard embeddings")
    create_embeddings(list(word2vec))

    print("Calculating T-SNE for word2vec model")
    x_coords, y_coords, labels, arr = calculate_t_sne()



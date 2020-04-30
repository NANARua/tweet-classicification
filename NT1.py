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
from clean_text import cleantext


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



def get_word2vec(sentences):
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






if __name__ == '__main__':
    input_dir = "/Users/nanarua/Desktop/FYP/data"
    save_dir = "/Users/nanarua/DesktopFYP/analysis"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Preprocessing raw data")
    raw_input_file = os.path.join(input_dir, "/Users/nanarua/Desktop/FYP/data/data_set.csv")
    filename = os.path.join(save_dir, "data.json")
    processed = try_load_or_process(filename, cleantext, raw_input_file)
    print("Unique sentences: " + str(len(processed)))

    print("Tokenizing sentences")
    filename = os.path.join(save_dir, "tokens.json")
    tokens = try_load_or_process(filename, tokenize_sentences, processed)

    print("Cleaning tokens")
    filename = os.path.join(save_dir, "cleaned.json")
    cleaned = try_load_or_process(filename, clean_sentences, tokens)

    print("Getting word frequencies")
    filename = os.path.join(save_dir, "frequencies.json")
    frequencies = try_load_or_process(filename, get_word_frequencies, cleaned)
    vocab_size = len(frequencies)
    print("Unique words: " + str(vocab_size))





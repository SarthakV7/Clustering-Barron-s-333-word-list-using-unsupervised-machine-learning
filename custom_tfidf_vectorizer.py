import pandas as pd
import numpy as np


class tfidf_vectorizer:
    def __init__(self):
        self.n_grams = []
        self.tf = []
        self.idf = []
        self.vocab = []
        self.tfidf_ = []
    def fit(self, data):
        uni_grams = set()
        bi_grams = set()
        for rows in data:
            words = rows.split(' ')
            for word_pair in zip(words, words[1:]):
                uni_grams.add(word_pair[0])
                bi_grams.add(word_pair)
            uni_grams.add(word_pair[1])
        self.n_grams = list(uni_grams.union(bi_grams))

    def transform(self, data):
        tf_ = pd.DataFrame(columns=[self.n_grams])
        idf_ = dict.fromkeys(self.n_grams, 0)
        idf_list = [1]*len(self.n_grams)
        for idx,rows in enumerate(data):
            words = rows.split(' ')
            tf = dict.fromkeys(self.n_grams, 0)
            for word_pair in zip(words, words[1:]):
                tf[word_pair] += 1
                tf[word_pair[0]] += 1
                idf_[word_pair] = 1
                idf_[word_pair[0]] += 1
            tf[word_pair[1]] += 1
            idf_[word_pair[1]] += 1
            idf_list += np.array(list(idf_.values()))
            vector = np.array(list(tf.values()))
            vector = vector/len(words)
            tf_.loc[idx] = vector
        # print(idf_list)
        idf_ = np.array([np.log(len(data)/term) for term in idf_list])
        # idf_ = np.fit_transform(idf_.reshape(1, -1))[0]
        tfidf_ = tf_.values*idf_
        self.tf = tf_
        self.idf = idf_
        return tfidf_

    def fit_transform(self, data):
        fit_ = fit(self, data)
        transform_ = transform(self, data)
        return transform_

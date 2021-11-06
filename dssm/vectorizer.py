# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
import pickle


class Vectorizer(CountVectorizer):

    def __init__(self, lang='zh', **kwargs):
        if lang == 'zh':
            super(Vectorizer, self).__init__(ngram_range=(1, 1), analyzer='char_wb', binary=False, **kwargs)
        elif lang == 'en':
            super(Vectorizer, self).__init__(ngram_range=(3, 3), analyzer='char_wb', binary=False, **kwargs)

    def fit(self, raw_documents):
        return super(Vectorizer, self).fit(raw_documents)

    def transform(self, raw_documents):
        return super(Vectorizer, self).transform(raw_documents)

    def partial_fit(self, data):
        if(hasattr(self, 'vocabulary_')):
            vocab = self.vocabulary_
        else:
            vocab = {}
        self.fit(data)
        vocab = list(set(vocab.keys()).union(set(self.vocabulary_)))
        self.vocabulary_ = {vocab[i]: i for i in range(len(vocab))}

    def load(self, path):
        with open(path, 'rb') as f:
            self.vocabulary_ = pickle.load(f)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.vocabulary_, f)

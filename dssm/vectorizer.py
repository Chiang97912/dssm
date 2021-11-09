# -*- coding: utf-8 -*-
from .utils import tokenize4zh
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import pickle


class Vectorizer(CountVectorizer):

    def __init__(self, lang=None, **kwargs):
        self.lang = lang
        super(Vectorizer, self).__init__(**kwargs)

    def preprocess(self, raw_documents):
        if self.lang == 'en':
            raw_documents = [' '.join(word_tokenize(raw_document)) for raw_document in raw_documents]
        elif self.lang == 'zh':
            raw_documents = [' '.join(tokenize4zh(raw_document)) for raw_document in raw_documents]
        return raw_documents

    def fit(self, raw_documents):
        raw_documents = self.preprocess(raw_documents)
        return super(Vectorizer, self).fit(raw_documents)

    def transform(self, raw_documents):
        raw_documents = self.preprocess(raw_documents)
        return super(Vectorizer, self).transform(raw_documents)

    def partial_fit(self, raw_documents):
        if(hasattr(self, 'vocabulary_')):
            vocab = self.vocabulary_
        else:
            vocab = {}

        raw_documents = self.preprocess(raw_documents)
        self.fit(raw_documents)
        vocab = list(set(vocab.keys()).union(set(self.vocabulary_)))
        self.vocabulary_ = {vocab[i]: i for i in range(len(vocab))}

    def load(self, path):
        with open(path, 'rb') as f:
            self.vocabulary_ = pickle.load(f)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.vocabulary_, f)

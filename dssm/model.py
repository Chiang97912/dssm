# -*- coding: utf-8 -*-
from .vectorizer import Vectorizer
from .dataset import Dataset
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import json
import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


logging.basicConfig(level=logging.INFO)

activation_dict = nn.ModuleDict([
    ['relu', nn.ReLU()],
    ['hardtanh', nn.Hardtanh()],
    ['relu6', nn.ReLU6()],
    ['sigmoid', nn.Sigmoid()],
    ['tanh', nn.Tanh()],
    ['softmax', nn.Softmax()],
    ['softmax2d', nn.Softmax2d()],
    ['logsoftmax', nn.LogSoftmax()],
    ['elu', nn.ELU()],
    ['selu', nn.SELU()],
    ['celu', nn.CELU()],
    ['hardshrink', nn.Hardshrink()],
    ['leakyrelu', nn.LeakyReLU()],
    ['logsigmoid', nn.LogSigmoid()],
    ['softplus', nn.Softplus()],
    ['softshrink', nn.Softshrink()],
    ['prelu', nn.PReLU()],
    ['softsign', nn.Softsign()],
    ['softmin', nn.Softmin()],
    ['tanhshrink', nn.Tanhshrink()],
    ['rrelu', nn.RReLU()],
    ['glu', nn.GLU()],
])


class DSSM(nn.Module):
    def __init__(self,
                 model_name_or_path=None,
                 gamma=None,
                 device='cpu',
                 lang='en',
                 mlp_num_layers=3,
                 mlp_num_units=300,
                 mlp_num_fan_out=128,
                 mlp_activation_func='tanh',
                 vocab_ngram_range=(1, 1),
                 vocab_analyzer='char_wb',
                 vocab_binary=False,
                 **kwargs):
        """Deep Structured Semantic Models

        The implementation of paper Learning Deep Structured Semantic Models for Web Search using Clickthrough Data

        Keyword Arguments:
            model_name_or_path {str} -- the path of model (default: {None})
            gamma {number} -- smoothing factor (default: {None})
            device {str} -- device (default: {'cpu'})
            lang {str} -- language (default: {'en'})
            mlp_num_layers {number} -- the number of mlp layer (default: {3})
            mlp_num_units {number} -- mlp hidden size (default: {300})
            mlp_num_fan_out {number} -- mlp output size (default: {128})
            mlp_activation_func {str} -- mlp activation function (default: {'tanh'})
            vocab_ngram_range {tuple} -- vocabulary ngram range (default: {(1,1)})
            vocab_analyzer {str} -- vocabulary analyzer (default: {'char_wb'})
            vocab_binary {str} -- If True, all non zero counts are set to 1. (default: {False})
        """
        super(DSSM, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.gamma = gamma
        self.device = device
        self.lang = lang
        self.mlp_num_layers = mlp_num_layers
        self.mlp_num_units = mlp_num_units
        self.mlp_num_fan_out = mlp_num_fan_out
        self.mlp_activation_func = mlp_activation_func
        self.vocab_ngram_range = vocab_ngram_range
        self.vocab_analyzer = vocab_analyzer
        self.vocab_binary = vocab_binary
        self.__dict__.update(kwargs)

        if not model_name_or_path:
            self.model_name_or_path = 'dssm-model'

        self.model_path = os.path.join(self.model_name_or_path, 'pytorch_model.bin')
        self.vocabulary_path = os.path.join(self.model_name_or_path, 'vocabulary.bin')
        self.config_path = os.path.join(self.model_name_or_path, 'config.json')
        self.load()

    def make_perceptron_layer(self, in_f, out_f, activation=None, dropout=None, batch_norm=False):
        perceptron = nn.Linear(in_f, out_f)
        nn.init.xavier_uniform_(perceptron.weight)
        sequential = [perceptron]
        if batch_norm:
            sequential.append(nn.BatchNorm1d(out_f))
        if dropout:
            sequential.append(nn.Dropout(dropout))
        if activation:
            sequential.append(activation)
        return nn.Sequential(*sequential)

    def make_mlp_layer(self, in_features, mlp_num_layers, mlp_num_units, mlp_num_fan_out):
        mlp_sizes = [
            in_features,
            *mlp_num_layers * [mlp_num_units],
            mlp_num_fan_out
        ]
        if self.mlp_activation_func in activation_dict:
            activation = activation_dict[self.mlp_activation_func]
        else:
            raise ValueError(
                'Could not interpret activation identifier: ' + str(self.mlp_activation_func)
            )
        mlp = [self.make_perceptron_layer(in_f, out_f, activation=activation, batch_norm=False) for in_f, out_f in zip(mlp_sizes, mlp_sizes[1:])]
        return nn.Sequential(*mlp)

    def build(self):

        self.mlp_left = self.make_mlp_layer(self.in_features,
                                            self.mlp_num_layers,
                                            self.mlp_num_units,
                                            self.mlp_num_fan_out)
        self.mlp_right = self.make_mlp_layer(self.in_features,
                                             self.mlp_num_layers,
                                             self.mlp_num_units,
                                             self.mlp_num_fan_out)

        self.learn_gamma = self.make_perceptron_layer(1, 1)

        self.to(self.device)

    def criterion(self, softmax_qp):
        # loss = -torch.log(torch.prod(softmax_qp))
        # loss = -torch.sum(torch.log(softmax_qp))
        loss = -torch.mean(torch.log(softmax_qp))
        return loss

    def forward(self, q, p, ns, norm=False):
        if norm:
            q = F.normalize(q, p=2, dim=-1)
            p = F.normalize(p, p=2, dim=-1)
            ns = F.normalize(ns, p=2, dim=-1)

        out_q = self.mlp_left(q)
        out_p = self.mlp_right(p)
        out_ns = [self.mlp_right(n) for n in ns.permute(1, 0, 2)]

        # Relevance measured by cosine similarity
        if not self.gamma:
            cos_qp = self.learn_gamma(F.cosine_similarity(out_q, out_p, dim=1).unsqueeze(1))
            cos_qns = [self.learn_gamma(F.cosine_similarity(out_q, out_n, dim=1).unsqueeze(1)) for out_n in out_ns]
        else:
            cos_qp = F.cosine_similarity(out_q, out_p, dim=1).unsqueeze(1) * self.gamma
            cos_qns = [F.cosine_similarity(out_q, out_n, dim=1).unsqueeze(1) * self.gamma for out_n in out_ns]

        cos_qds = [cos_qp, *cos_qns]
        cos_uni = torch.cat(cos_qds, 1)  # size: (batch_size, num_neg + 1)

        # posterior probability computed by softmax
        softmax_qp = F.softmax(cos_uni, dim=1)[:, 0]  # size: (batch_size)
        return softmax_qp

    def fit(self, queries, documents, lr=0.0001, epochs=20, batch_size=32, num_neg=4, train_size=0.8, weight_decay=0, lr_decay=1):
        """

        Train the model with query-document pairs.

        Arguments:
            queries {list} -- Query list
            documents {list} -- Document list

        Keyword Arguments:
            lr {number} -- Learning rate (default: {0.0001})
            epochs {number} -- Number of epochs for training (default: {20})
            batch_size {number} -- Batch size (default: {32})
            num_neg {number} -- Number of negative samples (default: {4})
            train_size {number} -- Training dataset ratio (default: {0.8})
            weight_decay {number} -- Weight decay for optimizer(L2 Regularization) (default: {0})
            lr_decay {number} -- Multiplicative factor of learning rate decay (default: {1})
        """

        self.vectorizer.fit(queries + documents)
        self.save()
        self.build()
        logging.info('features: %s' % self.in_features)

        queries = self.vectorizer.transform(queries).toarray()
        documents = self.vectorizer.transform(documents).toarray()
        query, doc_p, doc_ns = [], [], []
        total = len(queries)
        for q, d in zip(queries, documents):
            rand = [documents[i] for i in np.random.randint(total, size=num_neg)]
            query.append(q)
            doc_p.append(d)
            doc_ns.append(rand)

        total = len(query)
        train_size = int(total * train_size)
        eval_size = total - train_size
        dataset = Dataset(query, doc_p, doc_ns)
        train_dataset, eval_dataset = random_split(
            dataset=dataset,
            lengths=[train_size, eval_size],
            generator=torch.Generator().manual_seed(0)
        )
        train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_iterator = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
        batch_num = len(train_iterator)
        logging.info(self)
        weight_p, bias_p = [], []
        for name, p in self.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.Adam([{'params': weight_p, 'weight_decay': weight_decay},
                                      {'params': bias_p, 'weight_decay': 0}], lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=lr_decay)
        for epoch in range(epochs):
            self.train()
            scheduler.step()
            batch_idx = 0
            train_loss = []
            for q, p, ns in train_iterator:
                q = Variable(q).to(self.device)
                p = Variable(p).to(self.device)
                ns = Variable(ns).to(self.device)

                softmax_qp = self.forward(q, p, ns)
                loss = self.criterion(softmax_qp)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr = scheduler.get_lr()[0]
                loss = round(loss.data.item(), 5)
                train_loss.append(loss)
                logging.info(r'epoch: %s/%s, batch:%s/%s, lr: %s, loss: %s' %
                             (epoch + 1, epochs, batch_idx + 1, batch_num, lr, loss))
                batch_idx += 1

            self.eval()
            eval_loss = []
            for q, p, ns in eval_iterator:
                q = Variable(q).to(self.device)
                p = Variable(p).to(self.device)
                ns = Variable(ns).to(self.device)

                softmax_qp = self.forward(q, p, ns)
                loss = self.criterion(softmax_qp)
                loss = loss.data.item()
                eval_loss.append(loss)
            train_loss = round(np.average(train_loss), 5)
            eval_loss = round(np.average(eval_loss), 5)
            logging.info(r'epoch: %s/%s, lr: %s, train loss: %s, eval loss: %s' % (epoch + 1, epochs, lr, train_loss, eval_loss))
            os.makedirs(self.model_name_or_path, exist_ok=True)
            torch.save(self.state_dict(), self.model_path)

    def encode(self, texts):
        input_was_string = False
        if isinstance(texts, str):
            texts = [texts]
            input_was_string = True

        self.eval()
        vectors = self.vectorizer.transform(texts).toarray()
        with torch.no_grad():
            vectors = torch.FloatTensor(vectors).to(self.device)
        vectors = self.mlp_left(vectors).tolist()
        if input_was_string:
            vectors = vectors[0]
        return vectors

    def _save_config(self):
        config = {'lang': self.lang}
        for key, value in self.__dict__.items():
            if key.startswith('mlp') or key.startswith('vocab_'):
                config[key] = value
        os.makedirs(self.model_name_or_path, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    def _load_config(self):
        os.makedirs(self.model_name_or_path, exist_ok=True)
        with open(self.config_path, encoding='utf-8') as f:
            config = json.load(f)
        self.__dict__.update(config)

    def _save_vocab(self):
        os.makedirs(self.model_name_or_path, exist_ok=True)
        self.vectorizer.save(self.vocabulary_path)
        self.in_features = len(self.vectorizer.get_feature_names())

    def _load_vocab(self):
        self.vectorizer.load(self.vocabulary_path)
        self.in_features = len(self.vectorizer.get_feature_names())

    def save(self):
        self._save_vocab()
        self._save_config()

    def load(self):
        if os.path.exists(self.config_path):
            self._load_config()
        else:
            self.in_features = None

        config = {'lang': self.lang}
        for key, value in self.__dict__.items():
            if key.startswith('vocab_'):
                config[key.replace('vocab_', '')] = value
        self.vectorizer = Vectorizer(**config)

        if os.path.exists(self.vocabulary_path):
            self._load_vocab()
            self.build()
        if os.path.exists(self.model_path):
            self.load_state_dict(torch.load(self.model_path, map_location=self.device))

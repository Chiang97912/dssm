# -*- coding: utf-8 -*-
from torch.utils.data import Dataset as TorchDataset
import torch


class Dataset(TorchDataset):
    def __init__(self, query, doc_p, doc_ns):
        self.query = query
        self.doc_p = doc_p
        self.doc_ns = doc_ns

    def __len__(self):
        return len(self.query)

    def __getitem__(self, index):
        query = torch.FloatTensor(self.query[index])
        doc_p = torch.FloatTensor(self.doc_p[index])
        doc_ns = torch.FloatTensor(self.doc_ns[index])
        return query, doc_p, doc_ns

# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
from dssm.model import DSSM
import pandas as pd


df = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t')
question1 = []
question2 = []
for i, row in df.iterrows():
    if row['is_duplicate'] == 0:
        continue
    question1.append(row['question1'])
    question2.append(row['question2'])

model = DSSM('dssm-model',
             device='cpu',
             lang='en',
             vocab_ngram_range=(3, 3),
             vocab_analyzer='char_wb',
             vocab_binary=False)
model.fit(question1, question2, lr=0.0001)

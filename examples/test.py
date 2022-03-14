# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
from dssm.model import DSSM
from sklearn.metrics.pairwise import cosine_similarity


model = DSSM('dssm-model', device='cpu')
print('features:', model.in_features)

text_left = 'Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?'
text_right = 'I\'m a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?'
print(text_left)
print(text_right)
vectors = model.encode([text_left, text_right])
score = cosine_similarity([vectors[0]], [vectors[1]])
print('score:', score[0][0])


# DSSM
An industrial-grade implementation of the paper: [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://dl.acm.org/citation.cfm?id=2505665)


Latent semantic models, such as LSA, intend to map a query to its relevant documents at the semantic level where keyword-based matching often fails. DSSM project queries and documents into a common low-dimensional space where the relevance of a document given a query is readily computed as the distance between them.

This model can be used as a search engine that helps people find out their desired document even with searching a query that:
1. is abbreviation of the document words;
2. changed the order of the words in the document;
3. shortened words in the document;
4. has typos;
5. has spacing issues.



## Install

DSSM is dependent on PyTorch. Two ways to install DSSM:

**Install DSSM from Pypi:**

```
pip install dssm
```



**Install DSSM from the Github source:**

```
git clone https://github.com/Chiang97912/dssm.git
cd dssm
python setup.py install
```



## Usage

### Train

```python
from dssm.model import DSSM

queries = ['...']  # query list, words need to be segmented in advance, and tokens should be spliced with spaces.
documents = ['...']  # document list, words need to be segmented in advance, and tokens should be spliced with spaces.
model = DSSM('dssm-model', device='cuda:0', lang='en')
model.fit(queries, documents)
```



### Test

```python
from dssm.model import DSSM
from sklearn.metrics.pairwise import cosine_similarity

text_left = '...'
text_right = '...'
model = DSSM('dssm-model', device='cpu')
vectors = model.encode([text_left, text_right])
score = cosine_similarity([vectors[0]], [vectors[1]])
print(score)
```



## Dependencies

* `Python` version 3.6
* `Numpy` version 1.19.5
* `PyTorch` version 1.9.0


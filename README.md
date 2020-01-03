# PreNLP
[![PyPI](https://img.shields.io/pypi/v/prenlp.svg?style=flat-square&color=important)](https://pypi.org/project/prenlp/)
[![License](https://img.shields.io/github/license/lyeoni/prenlp?style=flat-square)](https://github.com/lyeoni/prenlp/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/lyeoni/prenlp?style=flat-square)](https://github.com/lyeoni/prenlp/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/lyeoni/prenlp?style=flat-square&color=blueviolet)](https://github.com/lyeoni/prenlp/network/members)

Preprocessing Library for Natural Language Processing

## Installation
### Requirements
- Python >= 3.6 
- Mecab morphological analyzer for Korean
  ```
  sh scripts/install_mecab.sh
  # only for Mac OS users, run the code below before run install_mecab.sh script.
  # export MACOSX_DEPLOYMENT_TARGET=10.10
  # CFLAGS='-stdlib=libc++' pip install konlpy
  ```
    
### With pip
prenlp can be installed using pip as follows:
```
pip install prenlp
```

## Usage

### Data

#### Dataset Loading
Popular datasets for NLP tasks are provided in prenlp.
- Language Modeling: WikiText-2, WikiText-103, WikiText-ko, NamuWiki-ko
- Sentiment Analysis: IMDb, NSMC

General use cases are as follows:

##### [WikiText-2 / WikiText-103](https://github.com/lyeoni/prenlp/blob/develop/prenlp/data/dataset/language_modeling.py)
```python
>>> wikitext2 = prenlp.data.WikiText2()
>>> len(wikitext2)
3
>>> train, valid, test = prenlp.data.WikiText2()
>>> train[0]
'= Valkyria Chronicles III ='
```

##### [IMDB](https://github.com/lyeoni/prenlp/blob/master/prenlp/data/dataset/sentiment.py)
```python
>>> imdb_train, imdb_test = prenlp.data.IMDB()
>>> imdb_train[0]
["Minor Spoilers<br /><br />Alison Parker (Cristina Raines) is a successful top model, living with the lawyer Michael Lerman (Chris Sarandon) in his apartment. She tried to commit ...", 'pos']
```

#### [Normalization](https://github.com/lyeoni/prenlp/blob/master/prenlp/data/normalizer.py)
Frequently used normalization functions for text pre-processing are provided in prenlp.
> url, HTML tag, emoticon, email, phone number, etc.

General use cases are as follows:
```python
>>> from prenlp.data import Normalizer
>>> normalizer = Normalizer()

>>> normalizer.normalize('Visit this link for more details: https://github.com/')
Visit this link for more details: [URL]

>>> normalizer.normalize('Use HTML with the desired attributes: <img src="cat.jpg" height="100" />')
Use HTML with the desired attributes: [TAG]

>>> normalizer.normalize('Hello 🤩, I love you 💓 !')
Hello [EMOJI], I love you [EMOJI] !

>>> normalizer.normalize('Contact me at lyeoni.g@gmail.com')
Contact me at [EMAIL]

>>> normalizer.normalize('Call +82 10-1234-5678')
Call [TEL]
```

### Tokenizer
Frequently used (subword) tokenizers for text pre-processing are provided in prenlp.
> SentencePiece, NLTKMosesTokenizer, Mecab

#### [SentencePiece](https://github.com/lyeoni/prenlp/blob/master/prenlp/tokenizer/tokenizer.py)
```python
>>> from prenlp.tokenizer import SentencePiece
>>> tokenizer = SentencePiece()
>>> tokenizer.train(input='corpus.txt', model_prefix='sentencepiece', vocab_size=10000)
>>> tokenizer.load('sentencepiece.model')
>>> tokenizer('Time is the most valuable thing a man can spend.') # same with tokenizer.tokenize('Time is the most valuable thing a man can spend.')
['▁Time', '▁is', '▁the', '▁most', '▁valuable', '▁thing', '▁a', '▁man', '▁can', '▁spend', '.']
>>> tokenizer.detokenize(['▁Time', '▁is', '▁the', '▁most', '▁valuable', '▁thing', '▁a', '▁man', '▁can', '▁spend', '.'])
Time is the most valuable thing a man can spend.
```

#### [Moses tokenizer](https://github.com/lyeoni/prenlp/blob/master/prenlp/tokenizer/tokenizer.py)
```python
>>> from prenlp.tokenizer import NLTKMosesTokenizer
>>> tokenizer = NLTKMosesTokenizer()
>>> tokenizer('Time is the most valuable thing a man can spend.')
['Time', 'is', 'the', 'most', 'valuable', 'thing', 'a', 'man', 'can', 'spend', '.']
```

## Author
- Hoyeon Lee @lyeoni
- email : lyeoni.g@gmail.com
- facebook : https://www.facebook.com/lyeoni.f

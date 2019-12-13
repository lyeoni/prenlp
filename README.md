# PreNLP
[![PyPI](https://img.shields.io/pypi/v/prenlp.svg?style=flat-square&color=important)](https://pypi.org/project/prenlp/)
[![License](https://img.shields.io/github/license/lyeoni/prenlp?style=flat-square)](https://github.com/lyeoni/prenlp/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/lyeoni/prenlp?style=flat-square)](https://github.com/lyeoni/prenlp/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/lyeoni/prenlp?style=flat-square&color=blueviolet)](https://github.com/lyeoni/prenlp/network/members)

Preprocessing Library for Natural Language Processing

## Installation
```
pip install prenlp
```

## Usage

### Data

#### [Dataset Loading](https://github.com/lyeoni/prenlp/blob/master/prenlp/data/dataset.py)
Popular datasets for NLP tasks are provided in prenlp.
- Text Classification: IMDB, NSMC 

General use cases (for IMDB) are as follows:
```python
>>> imdb_train, imdb_test = prenlp.data.IMDB()
>>> len(imdb_train), len(imdb_test)
25000 25000
>>> imdb_train[0]
("Minor Spoilers<br /><br />Alison Parker (Cristina Raines) is a successful top model, living with the lawyer Michael Lerman (Chris Sarandon) in his apartment. She tried to commit ...", 'pos')
```

#### [Normalization](https://github.com/lyeoni/prenlp/blob/master/prenlp/data/normalizer.py)
Frequently used normalization functions for text pre-processing are provided in prenlp.
> url, HTML tag, emoticon, email, phone number, etc.

General use cases (for Moses tokenizer) are as follows:
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

#### [Tokenizer](https://github.com/lyeoni/prenlp/blob/master/prenlp/tokenizer/tokenizer.py)
Frequently used tokenizers for text pre-processing are provided in prenlp.
> NLTKMosesTokenizer

General use cases (for Moses tokenizer) are as follows:
```python
>>> from prenlp.tokenizer import NLTKMosesTokenizer
>>> tokenizer = NLTKMosesTokenizer()
>>> tokenizer('PreNLP package provides a variety of text preprocessing tools.')
['PreNLP', 'package', 'provides', 'a', 'variety', 'of', 'text', 'preprocessing', 'tools', '.']
```

## Author
- Hoyeon Lee @lyeoni
- email : lyeoni.g@gmail.com
- facebook : https://www.facebook.com/lyeoni.f

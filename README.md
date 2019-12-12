# PreNLP
Preprocessing Library for Natural Language Processing

## Installation
```
pip install prenlp
```

## Usage

### Data

#### Normalization
```python
from prenlp.data.normalizer import Normalizer
normalizer = Normalizer()

normalizer.normalize('Visit this link for more details: https://github.com/')
# Visit this link for more details: [URL]

normalizer.normalize('Use HTML with the desired attributes: <img src="cat.jpg" height="100" />')
# Use HTML with the desired attributes: [TAG]

normalizer.normalize('Hello 🤩, I love you 💓 !')
# Hello [EMOJI], I love you [EMOJI] !

normalizer.normalize('Contact me at lyeoni.g@gmail.com')
# Contact me at [EMAIL]

normalizer.normalize('Call +82 10-1234-5678')
# Call [TEL]
```

## Author
- Hoyeon Lee @lyeoni
- email : lyeoni.g@gmail.com
- facebook : https://www.facebook.com/lyeoni.f

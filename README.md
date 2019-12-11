# PreNLP
Preprocessing Library for Natural Language Processing

## Installation
```
pip install prenlp
```

## Usage

### Data

#### Normalization
```
from prenlp.data.normalization import *

>>> url_normalize('Visit this link for more details: https://github.com/', repl='[URL]')
Visit this link for more details: [URL]

>>> tag_normalize('Use HTML with the desired attributes: <img src="cat.jpg" height="100" />', repl='[TAG]')
Use HTML with the desired attributes: [TAG]

>>> emoji_normalize('Hello 🤩, I love you 💓 !', repl='[EMOJI]')
Hello [EMOJI], I love you [EMOJI] !

>>> email_normalize('Contact me at lyeoni.g@gmail.com', repl='[EMAIL]')
Contact me at [EMAIL]

>>> tel_normalize('Call +82 10-1234-5678', repl='[TEL]')
Call [TEL]
```

## Author
- Hoyeon Lee @lyeoni
- email : lyeoni.g@gmail.com
- facebook : https://www.facebook.com/lyeoni.f

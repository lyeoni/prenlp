# Examples

## Text Classification

### fastText on IMDb
Based on the [`fasttext_imdb.py`](https://github.com/lyeoni/prenlp/blob/master/examples/fasttext_imdb.py).

The following example code trains fastText classification model on IMDb.
The code below has only 16 lines of code (except blank lines and comments).

```python
import fasttext
import prenlp
from prenlp.data import Normalizer
from prenlp.tokenizer import NLTKMosesTokenizer

# Data Preparation
imdb_train, imdb_test = prenlp.data.IMDB()

# Preprocessing
tokenizer = NLTKMosesTokenizer()
normalizer = Normalizer(url_repl=' ', tag_repl=' ', emoji_repl=' ', email_repl=' ', tel_repl=' ')

for dataset in [imdb_train, imdb_test]:
    for i, (text, label) in enumerate(dataset):
        dataset[i][0] = ' '.join(tokenizer(normalizer.normalize(text.strip()))) # both

prenlp.data.fasttext_transform(imdb_train, 'imdb.train')
prenlp.data.fasttext_transform(imdb_test, 'imdb.test')
         
# Train
model = fasttext.train_supervised(input='imdb.train', epoch=20)

# Evaluate
print(model.test('imdb.train'))
print(model.test('imdb.test'))

# Inference
print(model.predict(imdb_test[0][0]))
```

#### Comparisons with tokenizers
Below table shows the accuracy from various tokenizer.

|Tokenizer|Acc (train)|Acc (test)|
|-|-:|-:|
|-|0.9888|0.8763|
|[NLTKMosesTokenizer](https://github.com/lyeoni/prenlp/blob/master/examples/fasttext_imdb.py)|0.9641|**0.8858**|
|[SentencePiece](https://github.com/lyeoni/prenlp/blob/master/examples/fasttext_imdb_sentencepiece.py)|0.951|0.8753|
|NLTKMosesTokenizer -> SentencePiece|0.9609|0.879|

#### Comparisons with tokenizers on NSMC (Korean IMDb)
Below table shows the accuracy from various tokenizer.

|Tokenizer|Acc (train)|Acc (test)|
|-|-:|-:|
|-|0.9964|0.7827|
|[Mecab](https://github.com/lyeoni/prenlp/blob/master/examples/fasttext_nsmc.py)|0.8996|0.8461|
|[SentencePiece](https://github.com/lyeoni/prenlp/blob/master/examples/fasttext_nsmc_sentencepiece.py)|0.8780|**0.8484**|
|Mecab -> SentencePiece|0.8606|0.8415|
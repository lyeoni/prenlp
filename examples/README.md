# Examples

## Text Classification

### fastText on IMDb
Based on the [`fasttext_imdb.py`](https://github.com/lyeoni/prenlp/blob/master/examples/fasttext_imdb.py).

The following example code trains fastText classification model on IMDb.
The code below has only 17 lines of code (except blank lines and comments).

```python
import fasttext
import prenlp
from prenlp.data import Normalizer
from prenlp.tokenizer import NLTKMosesTokenizer

normalizer = Normalizer(emoji_repl=None)

# Data preparation
imdb_train, imdb_test = prenlp.data.IMDB()

# Preprocessing
tokenizer = NLTKMosesTokenizer()
for dataset in [imdb_train, imdb_test]:
    for i, (text, label) in enumerate(dataset):
        dataset[i][0] = ' '.join(tokenizer(normalizer.normalize(text.strip()))) # both
        # dataset[i][0] = text.strip() # original
        # dataset[i][0] = normalizer.normalize(text.strip()) # only normalization
        # dataset[i][0] = ' '.join(tokenizer(text.strip())) # only tokenization

prenlp.data.fasttext_transform(imdb_train, 'imdb.train')
prenlp.data.fasttext_transform(imdb_test, 'imdb.test')
         
# Train
model = fasttext.train_supervised(input='imdb.train', epoch=25)

# Evaluate
print(model.test('imdb.train'))
print(model.test('imdb.test'))

# Inference
print(imdb_test[0][0])
print(model.predict(imdb_test[0][0]))
```
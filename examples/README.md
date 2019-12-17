# Examples

## Text Classification

### fastText on IMDB
Based on the [`fasttext_imdb.py`]().

The following example code **trains fastText classification model on IMDB**.
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
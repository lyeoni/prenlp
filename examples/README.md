# Examples

## Setup input pipeline

### Building Vocabulary
You can easily download corpus using prenlp. Here, we use WikiText-103.
```shell
$ python -c "import prenlp; prenlp.data.WikiText103()"
$ ls .data/wikitext-103/
wiki.test.tokens  wiki.train.tokens  wiki.valid.tokens
```

Build Vocabulary based on WikiText-103 corpus, using sentencepiece subword tokenizer.
```shell
$ python vocab.py --corpus .data/wikitext-103/wiki.train.tokens --prefix wiki --tokenizer sentencepiece --vocab_size 16000
```

You may need to change below argument to build your own vocabulary with your corpus.
```shell
$ python vocab.py -h
usage: vocab.py [-h] --corpus CORPUS --prefix PREFIX [--tokenizer TOKENIZER]
                [--vocab_size VOCAB_SIZE]
                [--character_coverage CHARACTER_COVERAGE]
                [--model_type MODEL_TYPE]
                [--max_sentence_length MAX_SENTENCE_LENGTH]
                [--pad_token PAD_TOKEN] [--unk_token UNK_TOKEN]
                [--bos_token BOS_TOKEN] [--eos_token EOS_TOKEN]

optional arguments:
  -h, --help            show this help message and exit
  --corpus CORPUS       one-sentence-per-line corpus file
  --prefix PREFIX       output vocab(or sentencepiece model) name prefix
  --tokenizer TOKENIZER
                        tokenizer to tokenize input corpus. available:
                        sentencepiece, nltk_moses, mecab
  --vocab_size VOCAB_SIZE
                        the maximum size of the vocabulary
  --character_coverage CHARACTER_COVERAGE
                        amount of characters covered by the model, good
                        defaults are: 0.9995 for languages with rich character
                        set like Japanse or Chinese and 1.0 for other
                        languages with small character set
  --model_type MODEL_TYPE
                        sentencepiece model type. Choose from unigram, bpe,
                        char, or word
  --max_sentence_length MAX_SENTENCE_LENGTH
                        The maximum input sequence length
  --pad_token PAD_TOKEN
                        token that indicates padding
  --unk_token UNK_TOKEN
                        token that indicates unknown word
  --bos_token BOS_TOKEN
                        token that indicates beginning of sentence
  --eos_token EOS_TOKEN
                        token that indicates end of sentence
```

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
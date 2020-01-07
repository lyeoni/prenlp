import fasttext
import prenlp
from prenlp.data import Normalizer
from prenlp.tokenizer import SentencePiece

VOCAB_SIZE = 30000
normalizer = Normalizer(emoji_repl=None)

# Data preparation
imdb_train, imdb_test = prenlp.data.IMDB()

# Corpus preparation for training SentencePiece
corpus_path = 'corpus.txt'
with open(corpus_path, 'w', encoding='utf-8') as writer:
    wikitext2 = prenlp.data.WikiText2()
    for dataset in wikitext2:
        for text in dataset:
            writer.write(normalizer.normalize(text.strip())+'\n')

# Preprocessing
tokenizer = SentencePiece()
tokenizer.train(input=corpus_path, model_prefix='sentencepiece', vocab_size=VOCAB_SIZE)
tokenizer.load('sentencepiece.model')
for dataset in [imdb_train, imdb_test]:
    for i, (text, label) in enumerate(dataset):
        dataset[i][0] = ' '.join(tokenizer(normalizer.normalize(text.strip())))

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
import fasttext
import prenlp
from prenlp.data import Normalizer
from prenlp.tokenizer import SentencePiece

# Data preparation
imdb_train, imdb_test = prenlp.data.IMDB()

# Corpus preparation for training SentencePiece
corpus_path = 'corpus.txt'
with open(corpus_path, 'w', encoding='utf-8') as writer:
    for text, label in imdb_train:
        writer.write(text.strip()+'\n')

# Preprocessing
tokenizer = SentencePiece()
tokenizer.train(input=corpus_path, model_prefix='sentencepiece', vocab_size=10000)
tokenizer.load('sentencepiece.model')
normalizer = Normalizer(url_repl=' ', tag_repl=' ', emoji_repl=' ', email_repl=' ', tel_repl=' ')

for dataset in [imdb_train, imdb_test]:
    for i, (text, label) in enumerate(dataset):
        dataset[i][0] = ' '.join(tokenizer(normalizer.normalize(text.strip())))

prenlp.data.fasttext_transform(imdb_train, 'imdb.train')
prenlp.data.fasttext_transform(imdb_test, 'imdb.test')
         
# Train
model = fasttext.train_supervised(input='imdb.train', epoch=20)

# Evaluate
print(model.test('imdb.train'))
print(model.test('imdb.test'))

# Inference
print(model.predict(imdb_test[0][0]))
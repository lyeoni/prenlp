import fasttext
import prenlp
from prenlp.data import Normalizer
from prenlp.tokenizer import SentencePiece

# Data Preparation
nsmc_train, nsmc_test = prenlp.data.NSMC()

# Corpus preparation for training SentencePiece
corpus_path = 'corpus.txt'
with open(corpus_path, 'w', encoding='utf-8') as writer:
    for text, label in nsmc_train:
        writer.write(text.strip()+'\n')

# Preprocessing
tokenizer = SentencePiece()
tokenizer.train(input=corpus_path, model_prefix='sentencepiece', vocab_size=10000)
tokenizer.load('sentencepiece.model')
normalizer = Normalizer(url_repl=' ', tag_repl=' ', emoji_repl=None, email_repl=' ', tel_repl=' ')

for dataset in [nsmc_train, nsmc_test]:
    for i, (text, label) in enumerate(dataset):
        dataset[i][0] = ' '.join(tokenizer(normalizer.normalize(text.strip())))

prenlp.data.fasttext_transform(nsmc_train, 'nsmc.train')
prenlp.data.fasttext_transform(nsmc_test, 'nsmc.test')
         
# Train
model = fasttext.train_supervised(input='nsmc.train', epoch=20)

# Evaluate
print(model.test('nsmc.train'))
print(model.test('nsmc.test'))

# Inference
print(model.predict(nsmc_test[0][0]))
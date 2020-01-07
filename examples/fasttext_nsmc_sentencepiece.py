import fasttext
import prenlp
from prenlp.data import Normalizer
from prenlp.tokenizer import SentencePiece

VOCAB_SIZE = 30000
normalizer = Normalizer(emoji_repl=None)

# Data preparation
nsmc_train, nsmc_test = prenlp.data.NSMC()

# Corpus preparation for training SentencePiece 
corpus_path = 'corpus.txt'
with open(corpus_path, 'w', encoding='utf-8') as writer:
    wikitexko = prenlp.data.WikiTextKo()
    for text in wikitexko:
        writer.write(normalizer.normalize(text.strip())+'\n')

# Preprocessing
tokenizer = SentencePiece()
tokenizer.train(input=corpus_path, model_prefix='sentencepiece', vocab_size=VOCAB_SIZE)
tokenizer.load('sentencepiece.model')
for dataset in [nsmc_train, nsmc_test]:
    for i, (text, label) in enumerate(dataset):
        dataset[i][0] = ' '.join(tokenizer(normalizer.normalize(text.strip())))

prenlp.data.fasttext_transform(nsmc_train, 'nsmc.train')
prenlp.data.fasttext_transform(nsmc_test, 'nsmc.test')
         
# Train
model = fasttext.train_supervised(input='nsmc.train', epoch=25)

# Evaluate
print(model.test('nsmc.train'))
print(model.test('nsmc.test'))

# Inference
print(nsmc_test[0][0])
print(model.predict(nsmc_test[0][0]))
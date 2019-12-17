import fasttext
import prenlp
from prenlp.data import Normalizer
from prenlp.tokenizer import Mecab

# Data Preparation
nsmc_train, nsmc_test = prenlp.data.NSMC()

# Preprocessing
tokenizer = Mecab()
normalizer = Normalizer(url_repl=' ', tag_repl=' ', emoji_repl=' ', email_repl=' ', tel_repl=' ')

import 
# for dataset in [nsmc_train, nsmc_test]:
#     for i, (text, label) in enumerate(dataset):
#         # dataset[i][0] = ' '.join(tokenizer(normalizer.normalize(text.strip()))) # both
#         # dataset[i][0] = text.strip() # original
#         # dataset[i][0] = normalizer.normalize(text.strip()) # only normalization
#         # dataset[i][0] = ' '.join(tokenizer(text.strip())) # only tokenization

# prenlp.data.fasttext_transform(imdb_train, 'imdb.train')
# prenlp.data.fasttext_transform(imdb_test, 'imdb.test')
         
# # Train
# model = fasttext.train_supervised(input='imdb.train', epoch=20)

# # Evaluate
# print(model.test('imdb.train'))
# print(model.test('imdb.test'))

# # Inference
# print(imdb_test[0][0])
# print(model.predict(imdb_test[0][0]))
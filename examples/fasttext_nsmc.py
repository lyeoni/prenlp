import fasttext
import prenlp
from prenlp.data import Normalizer
from prenlp.tokenizer import Mecab

normalizer = Normalizer(emoji_repl=None)

# Data preparation
nsmc_train, nsmc_test = prenlp.data.NSMC()

# Preprocessing
tokenizer = Mecab()
for dataset in [nsmc_train, nsmc_test]:
    for i, (text, label) in enumerate(dataset):
        dataset[i][0] = ' '.join(tokenizer(normalizer.normalize(text.strip()))) # both
        # dataset[i][0] = text.strip() # original
        # dataset[i][0] = normalizer.normalize(text.strip()) # only normalization
        # dataset[i][0] = ' '.join(tokenizer(text.strip())) # only tokenization

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
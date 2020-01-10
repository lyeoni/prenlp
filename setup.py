import prenlp
import setuptools

with open('README.md', encoding='utf-8') as reader:
    long_description = reader.read()

setuptools.setup(
    name                            = 'prenlp',
    version                         = prenlp.__version__,
    author                          = prenlp.__author__,
    author_email                    = prenlp.__email__,
    description                     = 'Preprocessing Library for Natural Language Processing',
    long_description                = long_description,
    long_description_content_type   = 'text/markdown',
    url                             = 'https://github.com/lyeoni/prenlp',
    packages                        = setuptools.find_packages(),
    install_requires                = [
        'nltk==3.2.5', 'konlpy', 'sentencepiece',   # Tokenizer
        'fasttext',                                 # Model
        'ijson', 'pyunpack', 'patool'               # Utils
    ],
    package_data                    = {},
    keywords                        = [
        'nlp',
        'text-preprocessing'
    ],
    classifiers                     = [
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Apache Software License'
    ]
)
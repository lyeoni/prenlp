import setuptools

__version__ = '0.0.1'

setuptools.setup(
    name                = 'nlp-preprocessing',
    version             = __version__,
    author              = 'Hoyeon Lee',
    author_email        = 'lyeoni.g@gmail.com',
    url                 = 'https://github.com/lyeoni/nlp-preprocessing',
    install_requires    = [],
    packages            = setuptools.find_packages(),
    keywords            = 'NLP',
    python_requires     = '>=3.5',
    package_data        = {},
    zip_safe            = False,
    classifiers         = [
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Apache License'
    ]
)
from pathlib import Path

from .base import Dataset

class WikiText2(Dataset):
    """WikiText-2 word-level dataset for language modeling.
    
    From:
        Salesforce, https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
    
    License:
        Creative Commons Attribution-ShareAlike
    
    Args:
        root (str): path to the dataset's highest level directory
    
    Examples:
    >>> wikitext2 = prenlp.data.WikiText2()
    >>> len(wikitext2)
    3
    >>> train, valid, test = prenlp.data.WikiText2()
    >>> len(train), len(valid), len(test)
    (23767, 2461, 2891)
    >>> train[0]
    = Valkyria Chronicles III =
    """

    def __init__(self, root: str='.data'):
        self.url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
        self.root = Path(root)
        self.dirname = 'wikitext-2'
        
        self.skip_empty = True # Whether to skip the empty samples (only for WikiText)

        if not (self.root/self.dirname).exists():
            super(WikiText2, self)._download(to_path = self.root)
        
        super(WikiText2, self).__init__(self._get_data())

    def _get_data(self, train: str='wiki.train.tokens', valid: str='wiki.valid.tokens',
                  test: str='wiki.test.tokens') -> list:
        dataset = []
        for i, data in enumerate([train, valid, test]):
            filename = self.root/self.dirname/data
            with open(filename, 'r', encoding='utf-8') as reader:
                if self.skip_empty:                    
                    samples = [line.strip() for line in reader.readlines() if line.strip()]
                else:
                    samples = [line.strip() for line in reader.readlines()]
                dataset.append(samples)
        
        return dataset


class WikiText103(Dataset):
    """WikiText-103 word-level dataset for language modeling.
    
    From:
        Salesforce, https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
    
    License:
        Creative Commons Attribution-ShareAlike
    
    Args:
        root (str): path to the dataset's highest level directory
    
    Examples:
    >>> wikitext103 = prenlp.data.WikiText103()
    >>> len(wikitext103)
    3
    >>> train, valid, test = prenlp.data.WikiText103()
    >>> len(train), len(valid), len(test)
    (1165029, 2461, 2891)
    >>> train[0]
    = Valkyria Chronicles III =
    """

    def __init__(self, root: str='.data'):
        self.url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip'
        self.root = Path(root)
        self.dirname = 'wikitext-103'
        
        self.skip_empty = True # Whether to skip the empty samples (only for WikiText)

        if not (self.root/self.dirname).exists():
            super(WikiText103, self)._download(to_path = self.root)
        
        super(WikiText103, self).__init__(self._get_data())

    def _get_data(self, train: str='wiki.train.tokens', valid: str='wiki.valid.tokens',
                  test: str='wiki.test.tokens') -> list:
        dataset = []
        for i, data in enumerate([train, valid, test]):
            filename = self.root/self.dirname/data
            with open(filename, 'r', encoding='utf-8') as reader:
                if self.skip_empty:                    
                    samples = [line.strip() for line in reader.readlines() if line.strip()]
                else:
                    samples = [line.strip() for line in reader.readlines()]
                dataset.append(samples)
        
        return dataset
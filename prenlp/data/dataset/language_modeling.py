import os
import re
import json
import ijson
from pathlib import Path

from .base import Dataset
from ..utils import download_from_url
from ..normalizer import Normalizer

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


class WikiTextKo(Dataset):
    """Wikipedia database dump (Korean) for language modeling.

    From:
        Wikipedia, https://dumps.wikimedia.org/kowiki/
    
    Args:
        root (str): path to the dataset's highest level directory
    
    Examples:
    >>> wikitextko = prenlp.data.WikiTextKo()
    >>> len(wikitextko)
    2334771
    >>> wikitextko[0]
    '지미 카터'
    >>> wikitextko[1]
    '제임스 얼 "지미" 카터 주니어(, 1924년 10월 1일 ~ )는 민주당 출신 미국 39번째 대통령 (1977년 ~ 1981년)이다.'
    """

    def __init__(self, root: str='.data'):
        self.url = 'https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2'
        self.root = Path(root)
        self.dirname = 'wikitext-ko'
        
        # WikiExtractor
        self.url_wikiextractor = 'https://raw.githubusercontent.com/attardi/wikiextractor/master/WikiExtractor.py'
        self.wikiextractor = 'WikiExtractor.py'

        if not (self.root/self.dirname).exists():
            self._download(to_path = self.root)

        super(WikiTextKo, self).__init__(self._get_data())
    
    def _download(self, to_path: str) -> None:
        """Override method of 'Dataset' class.
        """
        download_filename = self.url.split('/')[-1]
        from_path = download_from_url(self.url, download_filename, to_path)
        
        # Extracts and cleans text from a Wikipedia database dump using WikiExtractor.
        # WikiExtractor: https://github.com/attardi/wikiextractor
        wikiextractor_path = download_from_url(self.url_wikiextractor, self.wikiextractor, to_path)
        os.system(f'python {self.root/self.wikiextractor} -o {to_path/self.dirname} --json {from_path}')
        
    def _get_data(self) -> list:
        dataset = []
        filenames = [str(filename) for filename in (self.root/self.dirname).glob('**/wiki_*')]
        for filename in sorted(filenames):
            with open(filename, 'r', encoding='utf-8') as reader:
                for line in reader.readlines(): # line = a document
                    text = json.loads(line)['text'].strip()
                    # split document into sentences(len > 0)
                    samples = list(filter(lambda x: len(x) > 0, text.split('\n')))
                    dataset += samples
                    # If sample is a document, use below code not above two lines.
                    # sample = '\n'.join(list(filter(lambda x: len(x) > 0, text.split('\n'))))
                    # dataset.append(sample)

        return dataset

class NamuWikiKo(Dataset):
    """NamuWiki database dump (Korean) for language modeling.

    From:
        NamuWiki, https://namu.wiki/w/%EB%82%98%EB%AC%B4%EC%9C%84%ED%82%A4:%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4%20%EB%8D%A4%ED%94%84
    
    Args:
        root (str): path to the dataset's highest level directory
    
    Examples:
    >>> namuwikiko = prenlp.data.NamuWikiKo()
    >>> len(namuwikiko)
    32362607
    >>> namuwikiko[0]
    (신 세계수의 미궁 2에서 뜬 !!아앗!!)
    >>> namuwikiko[1]
    세계수의 미궁 시리즈에 전통으로 등장하는 대사. 세계수의 미궁 2 제왕의 성배|2편 제왕의 성배부터 등장했으며, 훌륭한 사망 플래그의 예시이다.
    """

    def __init__(self, root: str='.data'):
        self.url = 'https://dataserver.xyz/wikidb/namuwiki190312.7z'
        self.root = Path(root)
        self.dirname = 'namuwiki_20190312.json'
        self.normalize_dirname = 'namuwiki_20190312'

        if not (self.root/self.dirname).exists():
            super(NamuWikiKo, self)._download(to_path = self.root)
        
        if not (self.root/self.normalize_dirname).exists():
            # Save normalized dataset
            with open(self.root/self.normalize_dirname, 'w', encoding='utf-8') as writer:
                dataset = self._get_data()
                for text in dataset:
                    writer.write(text+'\n')
            super(NamuWikiKo, self).__init__(dataset)
        else:
            with open(self.root/self.normalize_dirname, 'r', encoding='utf-8') as reader:
                super(NamuWikiKo, self).__init__(reader.readlines())
        
    def _get_data(self) -> list:
        dataset = []
        with open(self.root/self.dirname, 'r', encoding='utf-8') as jfile:
            for item in ijson.items(jfile, 'item'):
                text = self._normalize(item['text']).strip()
                # split document into sentences(len > 0)
                samples = list(filter(lambda x: len(x) > 0, text.split('\n')))
                dataset += samples
                # If sample is a document, use below code not above two lines.
                # sample = '\n'.join(list(filter(lambda x: len(x) > 0, text.split('\n'))))
                # dataset.append(sample)

        return dataset
    
    def _normalize(self, text: str, repl: str='') -> str:
        """Return the normalized string.
        """
        regexs = [
                # macro
                r'\[+(?:[iI]nclude|youtube|분류|목차|각주|파일).*\]+',
                r'\#redirect[ \t]*.*', # e.g. #redirect blah
                # markup
                r"'''",                 # bold
                r'~~(?!~).*?~~',        # deletion
                r'--(?!~).*?--',        # deletion
                r'\|\|.*\|\|',          # table
                r'\{\{\{.*?\}\}\}',     # plain text {{{blah}}}
                r'^(\{\{\{.*)',         # incomplete-plain text {{{blah
                r'^(\}\}\})',           # }}}
                r'^([ \t]+\*).*',       # unordered list (*)
                r'[ \t]1\..*',          # unordered list (1.)
                r'\|\|',                # quote (multiple)
                r'\{\{\|',              # quote (multiple)
                r'\|\}\}',              # quote (multiple)
                r'^\>',                 # quote (sinlge))
                r'width=',
                # special markup - should follow above markup
                r'(?:\[\[)\S*?\|',      # hyperlink with alias (open)
                r'(?:\[\[)',            # hyperlink (open)
                r'(?:\]\])',            # hyperlink (close)
                r'\[\*(.*?)\]'          # footnote. It should follow the hyperlink patterns.
                ]
        for regex in regexs:
            regex = re.compile(regex, re.MULTILINE)
            text = regex.sub(repl, text)
        
        normalizer = Normalizer(emoji_repl=None)
        text = normalizer.normalize(text)
        return text
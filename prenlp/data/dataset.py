import zipfile
import tarfile
import urllib.request
from pathlib import Path

class Dataset():
    """Abstract dataset class for dataset-like object, like list and array.
    All datasets(sub-classes) should inherit.

    Args:
        data (list): dataset like object
    """

    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class IMDB(Dataset):
    """IMDB review dataset for sentiment analysis.

    Args:
        root (str): Path to the dataset's highest level directory
    From:
        http://ai.stanford.edu/~amaas/data/sentiment/
    """

    def __init__(self, root='.data'):
        self.url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        self.dirname = 'aclImdb'

        self.root = Path(root)
        self.filename = self.url.split('/')[-1]
        self.filepath = self.root/self.filename

        if not self.filepath.exists():
            if not self.root.exists():
                self.root.mkdir()
            self._download()
                
        super(IMDB, self).__init__(self._get_data())

    def _download(self):
        """Download and unzip an online archive (.tar.gz).
        """
        urllib.request.urlretrieve(self.url, self.filepath)

        with tarfile.open(self.filepath, 'r:gz') as tgfile:
            for tarinfo in tgfile:
                tgfile.extract(tarinfo, self.root)

    def _get_data(self):
        dataset = []
        for i, data in enumerate(['train', 'test']):
            samples = []
            for label in ['pos', 'neg']:
                files = (self.root/self.dirname/data/label).glob('*.txt')
                for file in files:
                    with open(file, 'r', encoding='utf-8') as reader:
                        text = reader.readline()
                        sample = [text, label]
                        samples.append(sample)
            dataset.append(samples)
        
        return dataset

class NSMC(Dataset):
    """NSMC (Naver Sentiment Move Corpus) review dataset for sentiment analysis.
    Reviews were written in the Korean.

    Args:
        root (str): Path to the dataset's highest level directory
    From:
        https://github.com/e9t/nsmc
    """

    def __init__(self, root='.data'):
        self.url = 'https://github.com/e9t/nsmc/archive/master.zip'
        self.dirname = 'nsmc-master'

        self.root = Path(root)
        self.filename = self.url.split('/')[-1]
        self.filepath = self.root/self.filename
        
        if not self.filepath.exists():
            if not self.root.exists():
                self.root.mkdir()
            self._download()
        
        super(NSMC, self).__init__(self._get_data())

    def _download(self):
        """Download and unzip an online archive (.zip).
        """
        urllib.request.urlretrieve(self.url, self.filepath)

        with zipfile.ZipFile(self.filepath, 'r') as zfile:
            zfile.extractall(self.root)

    def _get_data(self):
        dataset = []
        for i, data in enumerate(['ratings_train.txt', 'ratings_test.txt']):
            samples = []
            file = self.root/self.dirname/data
            with open(file, 'r', encoding='utf-8') as reader:
                for line in reader.readlines()[1:]: # not include column names
                    _line = line.strip().split('\t')
                    text, label = _line[1], int(_line[2])
                    sample = [text, label]
                    samples.append(sample)
            dataset.append(samples)

        return dataset
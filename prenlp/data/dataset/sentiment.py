from pathlib import Path

from .base import Dataset

class IMDB(Dataset):
    """IMDb review dataset for sentiment analysis.

    From:
        http://ai.stanford.edu/~amaas/data/sentiment/
    
    Args:
        root (str): path to the dataset's highest level directory
    
    Examples:
    >>> imdb_train, imdb_test = prenlp.data.IMDB()
    >>> len(imdb_train), len(imdb_test)
    (25000, 25000)
    >>> imdb_train[0]
    ["Minor Spoilers<br /><br />Alison Parker (Cristina Raines) is a successful top model, living with the lawyer Michael Lerman (Chris Sarandon) in his apartment. She tried to commit suicide twice in the past: the first time, when she was a teenager and saw her father cheating her mother with two women in her home, and then when Michael's wife died. Since then, she left Christ and the Catholic Church behind. Alison wants to live alone in her own apartment and with the help of the real state agent Miss Logan (Ava Gardner), she finds a wonderful furnished old apartment in Brooklyn Heights for a reasonable rental. She sees a weird man in the window in the last floor of the building, and Miss Logan informs that he is Father Francis Matthew Halloran (John Carradine), a blinded priest who lives alone supported by the Catholic Church. Alison moves to her new place, and once there, she receives a visitor: her neighbor Charles Chazen (Burgess Meredith) welcomes her and introduces the new neighbors to her. Then, he invites Alison to his cat Jezebel's birthday party in the night. On the next day, weird things happen with Alison in her apartment and with her health. Alison looks for Miss Logan and is informed that she lives alone with the priest in the building. A further investigation shows that all the persons she knew in the party were dead criminals. Frightened with the situation, Alison embraces Christ again, while Michael investigates the creepy events. Alison realizes that she is living in the gateway to hell. <br /><br />Although underrated in IMDb User Rating, 'The Sentinel' is one of the best horror movies ever. I have seen this film at least six times, being the first time in the 70's, in the movie theater. In 07 September 2002, I bought the imported DVD and saw it again. Yesterday I saw this movie once more. Even after so many years, this film is still terrific. The creepy and lurid story frightens even in the present days. The cast is a constellation of stars and starlets. You can see many actors and actresses, who became famous, in the beginning of career. Fans of horror movie certainly worships 'The Sentinel', and I am one of them. My vote is nine.<br /><br />Title (Brazil): 'A Sentinela dos Malditos' ('The Sentinel of the Damned')<br /><br />Obs.: On 02 September 2007, I saw this movie again.", 'pos']
    """

    def __init__(self, root: str='.data'):
        self.url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
        self.root = Path(root)
        self.dirname = 'aclImdb'

        if not (self.root/self.dirname).exists():
            super(IMDB, self)._download(to_path = self.root)
                        
        super(IMDB, self).__init__(self._get_data())

    def _get_data(self, train: str='train', test: str='test') -> list:
        dataset = []
        for i, data in enumerate([train, test]):
            samples = []
            for label in ['pos', 'neg']:
                filenames = (self.root/self.dirname/data/label).glob('*.txt')
                for filename in filenames:
                    with open(filename, 'r', encoding='utf-8') as reader:
                        text = reader.readline().strip()
                        sample = [text, label]
                        samples.append(sample)
            dataset.append(samples)
        
        return dataset

class NSMC(Dataset):
    """NSMC (Naver Sentiment Move Corpus) review dataset for sentiment analysis.
    Reviews were written in the Korean.

    From:
        https://github.com/e9t/nsmc

    Args:
        root (str): path to the dataset's highest level directory
    
    Examples:
    >>> nsmc_train, nsmc_test = prenlp.data.NSMC()
    >>> len(nsmc_train), len(nsmc_test)
    (150000, 50000)
    >>> nsmc_train[0]
    ['아 더빙.. 진짜 짜증나네요 목소리', 0]
    """

    def __init__(self, root: str='.data'):
        self.url = 'https://github.com/e9t/nsmc/archive/master.zip'
        self.root = Path(root)
        self.dirname = 'nsmc-master'
        
        if not (self.root/self.dirname).exists():
            super(NSMC, self)._download(to_path = self.root)
        
        super(NSMC, self).__init__(self._get_data())

    def _get_data(self, train: str='ratings_train.txt', test: str='ratings_test.txt') -> list:
        dataset = []
        for i, data in enumerate([train, test]):
            samples = []
            filename = self.root/self.dirname/data
            with open(filename, 'r', encoding='utf-8') as reader:
                for line in reader.readlines()[1:]: # not include column names
                    line = line.strip().split('\t')
                    text, label = line[1], int(line[2])
                    samples.append([text, label])
            dataset.append(samples)

        return dataset
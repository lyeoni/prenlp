class NLTKMosesTokenizer:
    """Create the Moses Tokenizer implemented by in NLTK.

    From:
        https://www.nltk.org/_modules/nltk/tokenize/moses.html
    
    Examples:
    >>> tokenizer = prenlp.tokenizer.NLTKMosesTokenizer()
    >>> tokenizer('PreNLP package provides a variety of text preprocessing tools.')
    ['PreNLP', 'package', 'provides', 'a', 'variety', 'of', 'text', 'preprocessing', 'tools', '.']
    >>> tokenizer('Time is the most valuable thing a man can spend.')
    ['Time', 'is', 'the', 'most', 'valuable', 'thing', 'a', 'man', 'can', 'spend', '.']
    """

    def __init__(self):
        try:
            from nltk.tokenize.moses import MosesTokenizer
        except Exception as ex:
            import nltk
            nltk.download('perluniprops')
            nltk.download('nonbreaking_prefixes')
        self.tokenizer = MosesTokenizer()
    
    def __call__(self, text: str) -> list:
        return self._tokenize(text)

    def _tokenize(self, text: str) -> list:
        return self.tokenizer.tokenize(text, escape=False)

class Mecab:
    """Create the Mecab morphological analyzer.

    From:
        https://bitbucket.org/eunjeon/mecab-ko-dic/src

    Examples:
    >>> tokenizer = prenlp.tokenizer.Mecab()
    >>> tokenizer('자연어 처리 전처리 패키지 PreNLP 많이 사랑해주세요!')
    ['자연어', '처리', '전처리', '패키지', 'PreNLP', '많이', '사랑', '해', '주', '세요', '!']
    >>> tokenizer('모든 이야기에는 끝이 있지만, 인생에서의 모든 끝은 새로운 시작을  의미한다.')
    ['모든', '이야기', '에', '는', '끝', '이', '있', '지만', ',', '인생', '에서', '의', '모든', '끝', '은', '새로운', '시작', '을', '의미', '한다', '.']
    """

    def __init__(self):
        try:
            from konlpy.tag import Mecab
        except ImportError:
            raise ImportError(
                'Mecab is not installed. '
                'You can install Mecab with "sh scripts/install_mecab.sh" '
                'You can refer to the installation guide in https://github.com/lyeoni/prenlp/blob/master/scripts/install_mecab.sh or https://bitbucket.org/eunjeon/mecab-ko-dic/src')
        self.tokenizer = Mecab()
    
    def __call__(self, text: str) -> list:
        return self._tokenize(text)
    
    def _tokenize(self, text: str) -> list:
        return self.tokenizer.morphs(text)

class SentencePiece:
    """Create the SentencePiece subword tokenizer.

    From:
        https://github.com/google/sentencepiece
    
    Examples:
    >>> tokenizer = prenlp.tokenizer.SentencePiece()
    >>> tokenizer.train(input='corpus.txt', model_prefix='sentencepiece', vocab_size=10000)
    >>> tokenizer.load('sentencepiece.model')
    >>> tokenizer('Time is the most valuable thing a man can spend.')
    ['▁Time', '▁is', '▁the', '▁most', '▁valuable', '▁thing', '▁a', '▁man', '▁can', '▁spend', '.']
    >>> tokenizer.tokenize('Time is the most valuable thing a man can spend.')
    ['▁Time', '▁is', '▁the', '▁most', '▁valuable', '▁thing', '▁a', '▁man', '▁can', '▁spend', '.']
    >>> tokenizer.detokenize(['▁Time', '▁is', '▁the', '▁most', '▁valuable', '▁thing', '▁a', '▁man', '▁can', '▁spend', '.'])
    Time is the most valuable thing a man can spend.
    """

    def __init__(self, model: str = None):
        try:
            import sentencepiece
        except ImportError:
            raise ImportError(
                'SentencePiece is not installed. '
                'You can install Python binary package of SentencePiece with "pip install sentencepiece" '
                'You can refer to the official installation guide in https://github.com/google/sentencepiece')
        self.sentencepiece = sentencepiece
        self.processor = sentencepiece.SentencePieceProcessor()
        
        # Load model
        if model is not None:
            self.load(model)

    def __call__(self, text: str) -> list:
        return self.tokenize(text)
    
    def train(self, input: str, model_prefix: str, vocab_size: int, character_coverage: float = 1.0, model_type: str = 'bpe') -> None:
        """Train SentencePiece model.
        Args:
            input (str): one-sentence-per-line raw corpus file
            model_prefix (str): output model name prefix. <model_prefix>.model and <model_prefix>.vocab are generated
            vocab_size (int): vocabulary size 
            character_coverage (float): amount of characters covered by the model, 
                                        good defaults are: 0.9995 for languages with rich character set like Japanse or Chinese and 1.0 for other languages with small character set
            model_type (str): model type. Choose from bpe, unigram, char, or word.
                              The input sentence must be pretokenized when using word type
        """
        cmd = '--input={} --model_prefix={} --vocab_size={} --character_coverage={} --model_type={}'.format(
            input, model_prefix, vocab_size, character_coverage, model_type)
        self.sentencepiece.SentencePieceTrainer.Train(cmd)
    
    def load(self, model: str) -> None:
        """Load the pre-trained SentencePiece model.
        Args:
            model (str): pre-trained sentencepiece model
        """
        self.processor.Load(model)
    
    def tokenize(self, text: str) -> list:
        return self.processor.EncodeAsPieces(text)
    
    def detokenize(self, tokens: list) -> str:
        return self.processor.DecodePieces(tokens)
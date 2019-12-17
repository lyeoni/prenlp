class NLTKMosesTokenizer():
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
    
    def __call__(self, text: str):
        return self._tokenize(text)

    def _tokenize(self, text: str):
        return self.tokenizer.tokenize(text, escape=False)

class Mecab():
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
        except Exception as ex:
            pass

        self.tokenizer = Mecab()
    
    def __call__(self, text: str):
        return self._tokenize(text)
    
    def _tokenize(self, text: str):
        return self.tokenizer.morphs(text)
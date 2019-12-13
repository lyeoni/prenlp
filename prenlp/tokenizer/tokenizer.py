
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
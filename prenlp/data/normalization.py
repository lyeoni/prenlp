import re

def url_normalize(text, repl='[URL]'):
    """Return the string obtained by replacing all urls in 'text' by the replacement 'repl'.
    Args:
        text (str): text to be replaced
        repl (str): replace all urls in text with 'repl'
    """
    text = re.sub(r'(https?|ftp|www)\S+', repl, text)
    return text

def tag_normalize(text, repl=''):
    """Return the string obtained by replacing all HTML tags in 'text' by the replacement 'repl'.
    Args:
        text (str): text to be replaced
        repl (str): replace all urls in text with 'repl'
    """
    text = re.sub(r'<[^>]*>', repl, text)
    return text
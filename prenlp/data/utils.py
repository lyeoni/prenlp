
def fasttext_transform(data, filename: str, label_prefix: str ='__label__') -> None:
    """fastText style data transformation.
    Args:
        data (list, tuple): dataset-like object to be transformed. It should be in following format,
                            e.g. data[0] = (text, label)
        filename (str): filename of fasttext style output
        label_prefix (str): string, which is how fastText recognize what is a label.
    """
    with open(filename, 'w', encoding='utf-8') as writer:
        for _text, _label in data:
            writer.write('{prefix}{label} {text}\n'.format(
                prefix=label_prefix, label=_label, text=_text.strip()))
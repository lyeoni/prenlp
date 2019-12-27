import requests
import zipfile
import tarfile
from pathlib import Path

def fasttext_transform(data, filename: str, label_prefix: str='__label__') -> None:
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


def download_from_url(url: str, filename: str, root: str) -> Path:
    """Download file from url.
    Args:
        url (str): url of the file
        filename (str): filename to be downloaded
        root (str): directory used to store the file in, from url

    Returns:
        path to the downloaded files
    """
    root = Path(root)
    if not root.exists():
        root.mkdir()
    
    filepath = root/filename
    response = requests.get(url)
    with open(filepath, 'wb') as file:
        file.write(response.content)

    return filepath

def unzip_archive(from_path: str, to_path: str) -> Path:
    """Unzip archive.
    Args:
        from_path (str): path of the archive
        to_path (str): path to the directory of extracted files
    
    Returns:
        path to the directory of extracted files
    """
    extenstion = ''.join(Path(from_path).suffixes)
    if extenstion == '.zip':
        with zipfile.ZipFile(from_path, 'r') as zfile:
            zfile.extractall(to_path)
    elif extenstion == '.tar.gz' or '.tgz':
        with tarfile.open(from_path, 'r:gz') as tgfile:
            for tarinfo in tgfile:
                tgfile.extract(tarinfo, to_path)
    
    return Path(to_path)
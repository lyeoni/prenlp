from ..utils import download_from_url, unzip_archive

class Dataset:
    """Abstract dataset class for dataset-like object, like list and array.
    All datasets(sub-classes) should inherit.

    Args:
        data (list, array, tuple): dataset like object
    """

    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def _download(self, to_path: str) -> None:
        """Download and unzip an archive.
        Args:
            to_path (str): path to the directory of extracted files
        """
        download_filename = self.url.split('/')[-1]
        from_path = download_from_url(self.url, download_filename, self.root)

        unzip_archive(from_path, to_path)
import requests
from tqdm import tqdm
from os.path import basename

def pairwise(it):
    it = iter(it)
    while True:
        try:
            yield next(it), next(it)
        except StopIteration:
            # no more elements in the iterator
            return


def download_file(url: str, fname: str, chunk_size=1024, use_progress_bar = True):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    received = 0
    if use_progress_bar:
        with open(fname, 'wb') as file, tqdm(
            desc=basename(fname),
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)
                received = received+ size
    else:
        with open(fname, 'wb') as file:
           for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                received = received+ size

    if received != total:
        raise Exception("Failed to download '{}'".format(url))
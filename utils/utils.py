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

def download_file_without_progress_bar(url, save_path):
  try:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
      for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
    return True
  except requests.exceptions.RequestException as e:
    print(f"Error downloading {url}: {e}")
    return False

def download_file_with_progress_bar(url: str, fname: str, chunk_size=8192):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        with open(fname, 'wb') as file, tqdm(
            desc=basename(fname),
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=chunk_size,
        ) as bar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False

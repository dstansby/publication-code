import urllib.request
import os
import gzip
import pathlib
import shutil


def download_and_unpack(url):
    urllib.request.urlretrieve(url, fname(url))
    path = fname(url)
    with gzip.open(path, 'rb') as f_in:
        with open(path.stem, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(path)


def fname(url):
    return pathlib.Path(url.split('/')[-1])


def file_exists(url):
    return pathlib.Path(fname(url).stem).exists()


def read_gong_map_list():
    with open('gong_map_list.txt') as f:
        urls = f.readlines()
    urls = [x.strip() for x in urls]
    return urls


def gong_map_fnames():
    return [fname(url).stem for url in read_gong_map_list()]


def download_gong_maps():
    urls = read_gong_map_list()
    urls = [x for x in urls if not file_exists(x)]

    for url in urls:
        print(f'Downloading {url}')
        download_and_unpack(url)


if __name__ == '__main__':
    download_gong_maps()

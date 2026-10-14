import urllib.request


def download(url, dest, dp):
    dp.update(0)
    urllib.request.urlretrieve(url, dest, lambda nb, bs, fs: _pbhook(nb, bs, fs, dp))


def _pbhook(numblocks, blocksize, filesize, dp):
    percent = 100
    if filesize > 0:
        percent = int(min((numblocks * blocksize * 100) / filesize, 100))
    dp.update(percent)
    if dp.iscanceled():
        raise Exception('Canceled')

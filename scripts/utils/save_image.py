import numpy as np
import os
from time import time
import urllib
import multiprocessing
from progressbar import ProgressBar

def save_image(image: np.array, dir: str) -> bool:
    """_summary_

    Args:
        image (np.array): _description_
        dir (str): _description_

    Returns:
        bool: _description_
    """


    return True


import os
from time import time
import urllib
import multiprocessing
from progressbar import ProgressBar

img_dir = '/home/luke/datasets/sbu/images'

def get_sbu_urls():
    # read in urls
    f = open('SBU_captioned_photo_dataset_urls.txt', 'rb')
    urls = f.read().splitlines()
    return urls

def scrape_and_save(url, savepath):
    f = open(savepath,'wb')
    f.write(urllib.urlopen(url).read())
    f.close()

if __name__ == '__main__':
    """downloads the sbu dataset"""
    startidx = 439638

    urls = get_sbu_urls()
    urls = urls[startidx:]

    pool = multiprocessing.Pool(16)

    piccounter = 1

    def picdownloaded(arg):
        """Increment the piccounter variable for use in progressbar."""
        global piccounter
        piccounter += 1

    starttime = time()

    for i, url in enumerate(urls):
        name = 'SBU_%d.jpg' % (i + startidx)
        savepath = os.path.join(img_dir, name)
        pool.apply_async(scrape_and_save, (url, savepath), callback=picdownloaded)

    #Wait for now but will implement concurrent album downloads later
    totalpics = len(urls)
    with ProgressBar(maxval=totalpics) as progress:
        while piccounter < totalpics:
            progress.update(piccounter)

    duration = time() - starttime
    print("Download took: " + str(duration) + " seconds.\n")
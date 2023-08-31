import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

fileNames = ["out/_fpe_after", "out/_fpe_before"]

for fileName in fileNames:
    rawimage = Image.open(fileName + ".png")

    # Make DFT
    rawimageDFT = np.fft.fftshift(np.fft.fft2(rawimage))

    outimage = np.log(abs(rawimageDFT)+1)
    themax = np.max(outimage)
    outimage = 255 * outimage / themax

    new_p = Image.fromarray(outimage)
    if new_p.mode != 'RGB':
        new_p = new_p.convert('RGB')
    new_p.save(fileName + '.dft.png')

    # Make histogram
    histogram, bin_edges = np.histogram(rawimage, bins=256, range=(0, 255))

    plt.figure()
    plt.title("Histogram - " + fileName + ".png")
    plt.xlabel("pixel value")
    plt.ylabel("pixel count")
    plt.xlim([0.0, 255.0])  # <- named arguments do not work here

    plt.plot(bin_edges[0:-1], histogram)    

    plt.savefig(fileName + '.histogram.png')

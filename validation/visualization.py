import matplotlib.pyplot as plt
from tifffile import imread
import numpy
from skimage.metrics import structural_similarity

A='2912/00081.tiff'
B='00089.tiff'
C='gt/0008_200_gt.tiff'
D='/ddn/beamline/Fernando/upscaling/talitas/0008_200_lr.pkl'

a = numpy.array(imread(A))
b = numpy.array(imread(B))
c = numpy.array(imread(C))
d = numpy.fromfile(D, dtype='float32').reshape((2048, 2048))

a = ((a - a.min()) * 1.0000 / (a.max() - a.min()))
b = ((b - b.min()) * 1.0000 / (b.max() - b.min()))
c = ((c - c.min()) * 1.0000 / (c.max() - c.min()))
d = ((d - d.min()) * 1.0000 / (d.max() - d.min()))

print('First vs Ground truth {}'.format(structural_similarity(a, c)))
print('Current vs Ground truth {}'.format(structural_similarity(b, c)))
print('Bicubic vs Ground truth {}'.format(structural_similarity(d, c)))
print('Current vs Bicubic {}'.format(structural_similarity(b, d)))
print('Current vs First {}'.format(structural_similarity(b, a)))

plt.subplot(221)
plt.title('First epoch prediction')
plt.imshow(a)
plt.subplot(222)
plt.title('Current epoch prediction')
plt.imshow(b)
plt.subplot(223)
plt.title('Ground truth')
plt.imshow(c)
plt.subplot(224)
plt.title('Interpolation')
plt.imshow(d)
plt.show()

import matplotlib.pyplot as plt; from tifffile import imread; import numpy

A='1.tiff'
B='30.tiff'
C='0008_200_gt.tiff'
D='/ddn/beamline/Fernando/upscaling/talitas/0008_200_lr.pkl'

a = numpy.array(imread(A))
b = numpy.array(imread(B))
c = numpy.array(imread(C))
d = numpy.fromfile(D, dtype='float32').reshape((2048, 2048))

plt.subplot(221);plt.imshow(a);plt.subplot(222),plt.imshow(b); plt.subplot(223),plt.imshow(c);plt.subplot(224),plt.imshow(d);plt.show()

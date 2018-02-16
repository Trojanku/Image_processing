from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage import feature
from scipy import ndimage
import numpy as np


im = Image.open('lena.jpg')
width, height = im.size
pixels = list(im.getdata())
plt.imshow(im, cmap = plt.get_cmap('gray'))
plt.show()

#                                             RGB to gray

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread('lena.jpg')
szary = rgb2gray(img)
plt.imshow(szary, cmap = plt.get_cmap('gray'))
plt.show()

#                                             canny
edges = feature.canny(szary,sigma= 3)
plt.imshow(edges,cmap = 'gray')
plt.show()

#                                            histgoram

R = []
G = []
B = []
gray = []

for i in pixels:
    R.append(i[0])
    G.append(i[1])
    B.append(i[2])
    gray.append(round(i[0] * 0.299 + i[1] * 0.587 + i[2] * 0.114,0))


y_R = []
y_G = []
y_B = []
y_gray = []
x = []



for k in range(0,255):
    y_R.append(R.count(k))
    y_G.append(G.count(k))
    y_B.append(B.count(k))
    y_gray.append(gray.count(k))
    x.append(k)


plt.plot(x, y_R,'r','-')
plt.plot(x, y_G,'g', '-')
plt.plot(x, y_B,'b', '-')
plt.show()
plt.plot(x, y_gray,'gray', '-')
plt.show()


#                             krzyz robertsa


roberts_cross_v = np.array([[1, 0],
                               [0, -1]], dtype=np.double)

roberts_cross_h =  np.array([[0, 1],
                               [-1, 0]], dtype=np.double)


image = np.asarray(szary, dtype="int32")

vertical = ndimage.convolve( image, roberts_cross_v )
horizontal = ndimage.convolve( image, roberts_cross_h )

data = np.sqrt( np.square(horizontal) + np.square(vertical))

#img = Image.fromarray(np.asarray(np.clip(data, 0, 255), dtype="uint8"), "L")
#img.show()
plt.imshow(data, cmap = plt.get_cmap('gray'))
plt.show()

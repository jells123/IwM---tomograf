from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from skimage.transform import radon
from scipy import ndimage

def discrete_radon_transform(image, steps):
    R = np.zeros((steps, len(image)), dtype='float64')
    for s in range(steps):
        rotation = misc.imrotate(image, -s*180/steps).astype('float64')
        print(sum(rotation))
        R[:,s] = sum(rotation)
    return R

def getCirclePoint(radius, angle, x_origin, y_origin, degrees=True):
    if degrees:
        angle = math.radians(angle)
    x = x_origin + radius * math.cos(angle)
    y = y_origin + radius * math.sin(angle)
    return x, y

def getBresenhamLine(x1, y1, x2, y2):
    result = []
    # https://gist.github.com/bert/1085538
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    dx = np.abs(x2 - x1)
    dy = -1 * np.abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy
    while 1:
        result.append([x1, y1])
        if x2 == x1 and y2 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x1 += sx
        if e2 <= dx:
            err += dx; y1 += sy
    return result

# Read image as 64bit float gray scale
image = misc.imread('shepplogan3.png', flatten=True).astype('float64')

print(image.shape)

alpha = 0.75 #obrót tomografu
n = image.shape[0] #number of emiters
l = 180. #rozpiętość kątowa (?)

if image.shape[0] != image.shape[1]:
    print("Error! Image is not square.")

cx, cy = np.float(image.shape[0]/2), np.float(image.shape[1]/2)
radius = np.float(image.shape[0]/2)

angles  = list(np.arange(0., 180., alpha, dtype=np.float32))
emiters = list(np.linspace(-l/2, l/2, n, dtype=np.float32))

dist = l/n #angle distance between emiters
sinogramData = np.ndarray(shape=(len(angles), len(emiters)), dtype = np.float32)

for a, angle in enumerate(angles):

    projection = np.zeros(shape=(image.shape[0]), dtype=np.float32)

    for idx, e in enumerate(emiters):

        x1, y1 = getCirclePoint(radius, angle + e, cx, cy)
        x2, y2 = getCirclePoint(radius, angle + 180. - e, cx, cy)

        pixels = getBresenhamLine(x2, y2, x1, y1)
        remaining = int((len(projection) - len(pixels)) / 2)

        sum = 0
        for p, px in enumerate(pixels):
            try:
                sum += image[px[0]][px[1]]
                projection[p+remaining] += image[px[0]][px[1]]
            except IndexError:
                pass

        sinogramData[a][idx] = sum
        #if sum == 0:
        #    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
        #sinogramData[a][idx] = sum

    #projection /= image.shape[0]
    #sinogramData[a] = projection
    print("Done! -> ", angle)

sinogram = np.zeros(shape=(len(angles), image.shape[0]), dtype=np.float32)


'''
off = 0
if n%2 == 0:
    off = dist/2

i = 0
while i < n/2:

    x, y = getCirclePoint(radius, angle + i*dist + off, cx, cy)
    x2, y2 = getCirclePoint(radius, angle+180 - i*dist - off, cx, cy)
    cv2.line(image, (int(x), int(y)), (int(x2), int(y2)), (255,255,255), 1)

    x, y = getCirclePoint(radius, angle - i*dist - off, cx, cy)
    x2, y2 = getCirclePoint(radius, angle+180 + i*dist + off, cx, cy)
    cv2.line(image, (int(x), int(y)), (int(x2), int(y2)), (255,255,255), 1)

    i += 1
'''
if sinogram.sum() != 0:
    sinogram /= sinogram.sum()

'''
Równanie okręgu:
(x-a)**2 + (y-b)**2 = r**2
x = cx + r * cos(a)
y = cy + r * sin(a)
'''

xcenter, ycenter = np.float(image.shape[0]/2), np.float(image.shape[1]/2)
#cv2.circle(image, (int(cx), int(cy)), int(radius), (255,255,255), 2)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)
correctSinogram = radon(image, theta=theta, circle=True)

# Plot the original and the radon transformed image
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2), plt.imshow(ndimage.rotate(sinogramData, -90), cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3), plt.imshow(correctSinogram, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
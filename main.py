from scipy.fftpack import fft, ifft, fftfreq, rfftfreq, rfft, irfft
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.transform import radon
from scipy import ndimage
import cv2


def discrete_radon_transform(image, steps):
    R = np.zeros((steps, len(image)), dtype='float64')
    for s in range(steps):
        rotation = misc.imrotate(image, -s*180/steps).astype('float64')
        R[:,s] = sum(rotation)
    return R

def getCirclePoint(radius, angle, x_origin, y_origin, degrees=True):
    if degrees:
        angle = math.radians(angle)
    x = x_origin + radius * math.cos(angle)
    y = y_origin + radius * math.sin(angle)
    return x, y

def getBresenhamLine(x1, y1, x2, y2, max):
    result = []

    # https://gist.github.com/bert/1085538
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    dx = np.abs(x2 - x1)
    dy = -1 * np.abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy
    while 1:
        if max not in [x1, y1]:
            result.append([x1, y1])
        if x2 == x1 and y2 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x1 += sx
        if e2 <= dx:
            err += dx; y1 += sy

    return result

def getSquarePoints(points, squarePoints):
    result = []
    for p in points:
        if squarePoints[0] <= p[0] <= squarePoints[1] and squarePoints[2] <= p[1] <= squarePoints[3]:
            result.append(p)
    return result

# Read image as 64bit float gray scale
image = misc.imread('minilogan.png', flatten=True).astype('float64')
print(image.shape)

alpha = 180. / image.shape[0] #obrót tomografu
n = image.shape[0] #number of emiters
l = 180. #rozpiętość kątowa (?)

if image.shape[0] != image.shape[1]:
    print("Error! Image is not square.")

cx, cy = np.float(image.shape[0]/2), np.float(image.shape[1]/2)
radius = np.float(image.shape[0]/2)
square_size = image.shape[0]

angles  = list(np.arange(0., 180., alpha, dtype=np.float32))
emiters = list(np.linspace(-l/2, l/2, n, dtype=np.float32))

dist = l/n #angle distance between emiters
sinogramData = np.ndarray(shape=(len(angles), len(emiters)), dtype = np.float32)
image = np.array(image, dtype=np.float32)

width_1d = len(sinogramData[0])
circlePoints = {}
for a, angle in enumerate(angles):
    for idx, e in enumerate(emiters):

        x1, y1 = getCirclePoint(radius, angle + e, cx, cy)
        x2, y2 = getCirclePoint(radius, angle + 180. - e, cx, cy)
        circlePoints[(angle, e)] = [x1, y1, x2, y2]

        pixels = getBresenhamLine(x1, y1, x2, y2, square_size)
        sum = np.sum(list(map(lambda px: image[px[0]][px[1]], pixels)))
        sinogramData[a][idx] = sum

    get_row = np.array(sinogramData[a], dtype=np.float32)
    f = rfftfreq(width_1d)
    ramp_filter = np.abs(f) * 2
    projection = rfft(get_row, axis=0) * ramp_filter
    get_row = np.real(irfft(projection, axis=0))

    for i in range(len(get_row)):
        sinogramData[a][i] = get_row[i]

print("First iter done!")
newImage = np.zeros(shape = image.shape, dtype = np.float32)
border = image.shape[0] - 1

'''
closest_two = 2 ** np.ceil(np.log2(2 * sinogramData.shape[0]))
freqs_count = max(64, int(closest_two)) #nie może być mnie niż 64
pad_width = ((0, freqs_count - sinogramData.shape[0]), (0, 0))
sinogram_padded = np.pad(sinogramData, pad_width, mode='constant', constant_values=0)

f = fftfreq(freqs_count).reshape(-1, 1)
omega = 2 * np.pi * f
ramp_filter = 2 * np.abs(f)

projection = fft(sinogram_padded, axis=0) * ramp_filter
filtered_sgram = np.real(ifft(projection, axis=0))

sinogramData = filtered_sgram[:sinogramData.shape[0], :]
'''

for aa, angle in enumerate(angles):
    for idx, e in enumerate(emiters):

        x1, y1, x2, y2 = circlePoints[(angle, e)]
        if int(x1) != int(x2) and int(y1) != int(y2):
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1

            x1, y1 = 0, b
            x2, y2 = border, a*border + b

            x3, y3 = -b/a, 0
            x4, y4 = (border - b)/a, border

            points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            points = list(filter(lambda x : 0 <= x[0] <= border and 0 <= x[1] <= border, points))
            x1, y1 = points[0]
            x2, y2 = points[1]

        elif int(x1) == int(x2):
            y1 = 0
            y2 = image.shape[0] - 1
        elif int(y1) == int(y2):
            x1 = 0
            x2 = image.shape[0] - 1

        pixels = getBresenhamLine(x1, y1, x2, y2, square_size)
        if len(pixels) == 0:
            print("#PRZYPAŁ") #jeden przypadek dzielenia przez zero, lol
            addValue = 0
        else:
            addValue = sinogramData[aa][idx] / len(pixels)

        newImageFlat = np.reshape(newImage, newshape=-1)
        newPixels = np.array(list(map(lambda x: x[1] + x[0] * image.shape[0], pixels)), dtype=np.uint32)
        newImageFlat[newPixels] += addValue
        newImage = np.reshape(newImageFlat, newshape=image.shape)


# sinogramData /= sinogramData.sum()
# sinogramData *= 255
# sinogramData /= np.max(sinogramData)
print(newImage[5][5])
xcenter, ycenter = np.float(image.shape[0]/2), np.float(image.shape[1]/2)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)
correctSinogram = radon(image, theta=theta, circle=True)

# Plot the original and the radon transformed image
plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(ndimage.rotate(sinogramData, -90), cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(newImage, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(correctSinogram, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
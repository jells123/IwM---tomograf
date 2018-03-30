import math
import operator
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from scipy import misc
from scipy import ndimage
from skimage.transform import radon
from sklearn.metrics import mean_squared_error

'''
!!!
Zliczać ile linii przeszło przez dany piksel.
Uśrednić wynik w danym pikselu przez liczbę linii, które przez niego przeszły
- jakie to daje rezultaty?
'''

class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.waiter = QLabel("Wait...")
        self.name = QLabel("")
        self.pic = QLabel(self)
        self.file = ""
        self.totalA = {}
        self.totalN = {}
        self.totalL = {}
        self.btn2 = QPushButton('Do Tomography', self)
        self.spin1 = QDoubleSpinBox(self)
        self.spin2 = QSpinBox(self)
        self.spin3 = QDoubleSpinBox(self)
        self.b = QPushButton('Statistics(alpha)', self)
        self.b1 = QPushButton('Statistics(n)', self)
        self.b2 = QPushButton('Statistics(l)', self)

        self.doCountLines = False
        self.czekBoks = QCheckBox("Normalize by lines count")

        self.initUI()

    def initUI(self):
        self.setGeometry(300, 300, 700, 710)
        self.setWindowTitle('Tomograf')
        self.setWindowIcon(QIcon('minilogan.png'))

        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)

        btn = QPushButton('Browse Files', self)
        btn.resize(btn.sizeHint())
        btn.clicked.connect(self.browseFiles)

        self.btn2.resize(self.btn2.sizeHint())
        self.btn2.clicked.connect(self.tomograph)
        self.btn2.setEnabled(False)

        title = QLabel('Select file:')
        title.setAlignment(Qt.AlignCenter)
        self.name.setAlignment(Qt.AlignCenter)
        self.waiter.setAlignment(Qt.AlignCenter)

        alpha_text = QLabel('Adjust alpha:')
        alpha_text.setAlignment(Qt.AlignCenter)
        n_text = QLabel('Adjust n:')
        n_text.setAlignment(Qt.AlignCenter)
        l_text = QLabel('Adjust l:')
        l_text.setAlignment(Qt.AlignCenter)

        self.spin1.setMinimum(0.5)
        self.spin1.setDecimals(3)
        self.spin1.setSingleStep(0.001)

        self.spin2.setMinimum(1)
        self.spin2.setSingleStep(1)
        self.spin2.setMaximum(999)

        self.spin3.setMinimum(1)
        self.spin3.setSingleStep(0.1)
        self.spin3.setMaximum(180.0)

        self.b.clicked.connect(self.funAlpha)
        self.b.setEnabled(False)
        self.b1.clicked.connect(self.funN)
        self.b1.setEnabled(False)
        self.b2.clicked.connect(self.funL)
        self.b2.setEnabled(False)

        grid = QGridLayout()
        grid.setSpacing(10)

        self.czekBoks.setChecked(False)
        self.czekBoks.stateChanged.connect(lambda: self.btnstate(self.czekBoks))

        grid.addWidget(title, 1, 0, 1, 1)
        grid.addWidget(btn, 1, 1, 1, 1)
        grid.addWidget(self.btn2, 2, 1, 1, 1)
        grid.addWidget(self.waiter, 2, 2, 1, 1)
        grid.addWidget(self.name, 1, 2, 1, 1)
        grid.addWidget(self.pic, 3, 0, 3, 3)
        grid.addWidget(alpha_text, 7, 0, 1, 1)
        grid.addWidget(n_text, 7, 1, 1, 1)
        grid.addWidget(l_text, 7, 2, 1, 1)
        grid.addWidget(self.spin1, 8, 0, 1, 1)
        grid.addWidget(self.spin2, 8, 1, 1, 1)
        grid.addWidget(self.spin3, 8, 2, 1, 1)
        grid.addWidget(self.b, 9, 0, 1, 1)
        grid.addWidget(self.b1, 9, 1, 1, 1)
        grid.addWidget(self.b2, 9, 2, 1, 1)
        grid.addWidget(self.czekBoks, 10, 0, 1, 3)

        self.waiter.hide()
        self.setLayout(grid)

        self.center()
        self.show()

    def btnstate(self, b):
        if b.isChecked() == True:
            self.doCountLines = True
        else:
            self.doCountLines = False

    @pyqtSlot()
    def browseFiles(self):
        filter = "Png File (*.png)"
        fileIn = QFileDialog.getOpenFileName(self, "Select Png File", "./example", filter)
        self.file = str(fileIn).split("'")[1]
        if self.file == '':
            return
        self.name.setText(self.file.split("/")[-1])
        self.name.show()
        self.btn2.setEnabled(True)

        a, b, c = start(self.file)
        self.spin1.setValue(a)
        self.spin2.setValue(b)
        self.spin3.setValue(c)

        self.update()

    @pyqtSlot()
    def tomograph(self):
        self.waiter.show()
        print(self.waiter.isVisible())  # nie dziala, lol...
        print(self.waiter.text())
        # self.update()
        n = int(self.spin2.text())
        if n % 2 == 0:
            n += 1  # hehe programowanie

        y = doTomography(self.file, float(self.spin1.text().replace(',', '.')), n,
                     float(self.spin3.text().replace(',', '.')), self.doCountLines)
        self.pic.setPixmap(QPixmap(os.getcwd() + "/result.png").scaledToHeight(500))
        self.pic.show()
        self.waiter.hide()
        self.totalA[float(self.spin1.text().replace(',', '.'))] = y
        self.totalN[int(self.spin2.text())] = y
        self.totalL[float(self.spin3.text().replace(',', '.'))] = y
        # self.update()
        self.b.setEnabled(True)
        self.b1.setEnabled(True)
        self.b2.setEnabled(True)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    @pyqtSlot()
    def funAlpha(self):
        tmp = sorted(self.totalA.items(), key=operator.itemgetter(0))
        tmp = np.asarray(tmp, dtype=np.float32)
        plt.figure()
        plt.plot(tmp[:, 0], tmp[:, 1])
        plt.show()

    @pyqtSlot()
    def funN(self):
        tmp = sorted(self.totalN.items(), key=operator.itemgetter(0))
        tmp = np.asarray(tmp, dtype=np.float32)
        plt.figure()
        plt.plot(tmp[:, 0], tmp[:, 1])
        plt.show()

    @pyqtSlot()
    def funL(self):
        tmp = sorted(self.totalL.items(), key=operator.itemgetter(0))
        tmp = np.asarray(tmp, dtype=np.float32)
        plt.figure()
        plt.plot(tmp[:, 0], tmp[:, 1])
        plt.show()


def discrete_radon_transform(image, steps):
    R = np.zeros((steps, len(image)), dtype='float64')
    for s in range(steps):
        rotation = misc.imrotate(image, -s * 180 / steps).astype('float64')
        R[:, s] = sum(rotation)
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
            err += dy;
            x1 += sx
        if e2 <= dx:
            err += dx;
            y1 += sy

    return result


def getSquarePoints(points, squarePoints):
    result = []
    for p in points:
        if squarePoints[0] <= p[0] <= squarePoints[1] and squarePoints[2] <= p[1] <= squarePoints[3]:
            result.append(p)
    return result


def start(file):
    image = misc.imread(file, flatten=True).astype('float64')
    return math.floor((180. / image.shape[0])*1000)/1000, image.shape[0], 100.


def RMSE(image, newImage):
    return np.sqrt(mean_squared_error(image, newImage))


def normalize(img, originalMax):
    for i in range(len(img)):
        for j in range(len(img[i])):
            img[i, j] = img[i, j] / originalMax * 255

    return img


def normalize2(img):
    max_val = np.max(img)
    min_val = np.min(img)
    print(min_val, max_val)

    max_val += np.abs(min_val)
    for i in range(len(img)):
        for j in range(len(img)):
            img[i, j] = (img[i, j] + np.abs(min_val)) / max_val

    print(np.min(img), np.max(img))
    return img


def cutDownValues(img, cutDownValue=35):
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i, j] > cutDownValue:
                img[i, j] = img[i, j] - cutDownValue
            else:
                img[i, j] = 0
    return img


def plotTotal(totalX, totalY):
    plt.figure()
    plt.plot(totalX, totalY)
    plt.show()


def kernel(x):
    if x == 0:
        return 1
    elif x % 2 == 0:
        return 0
    else:
        return (-4/math.pi**2)/x**2


def getFilterKernel(len):
    if len % 2 != 1:
        print("Podano parzystą wielkość [kernel]")
        return
    kernel = np.zeros(len, dtype=np.float32)
    mid = int(len/2)
    kernel[mid] = 1
    for i in range(1, len-mid):
        if i % 2 == 0:
            kernel[mid+i] = 0
            kernel[mid-i] = 0
        elif i % 2 != 0:
            kernel[mid+i] = (-4/(np.pi**2)) / i**2
            kernel[mid-i] = (-4/(np.pi**2)) / i**2
    return kernel


def doTomography(file, alpha, n, l, doCountLines=False):
    # Read image as 64bit float gray scale
    image = misc.imread(file, flatten=True).astype('float64')
    print(image.shape)

    if image.shape[0] != image.shape[1]:
        print("Error! Image is not square.")

    cx, cy = np.float(image.shape[0] / 2), np.float(image.shape[1] / 2)
    radius = np.float(image.shape[0] / 2)
    square_size = image.shape[0]

    angles = list(np.arange(0., 180., alpha, dtype=np.float32))
    emiters = list(np.linspace(-l / 2, l / 2, n, dtype=np.float32))

    dist = l / n  # angle distance between emiters
    sinogramData = np.ndarray(shape=(len(angles), len(emiters)), dtype=np.float32)
    image = np.array(image, dtype=np.float32)

    width_1d = len(sinogramData[0])
    circlePoints = {}
    kernel = getFilterKernel(width_1d)

    for a, angle in enumerate(angles):
        for idx, e in enumerate(emiters):
            x1, y1 = getCirclePoint(radius, angle + e, cx, cy)
            x2, y2 = getCirclePoint(radius, angle + 180. - e, cx, cy)
            circlePoints[(angle, e)] = [x1, y1, x2, y2]

            pixels = getBresenhamLine(x1, y1, x2, y2, square_size)
            sum = np.sum(list(map(lambda px: image[px[0]][px[1]], pixels)))
            sinogramData[a][idx] = sum

        get_row = np.array(sinogramData[a], dtype=np.float32)
        row_filtered = np.convolve(get_row, kernel, mode='same')
        sinogramData[a, :] = row_filtered

    print("First iter done!")
    newImage = np.zeros(shape=image.shape, dtype=np.float32)
    countLines = np.zeros(shape=image.shape, dtype=np.float32)
    border = image.shape[0] - 1

    x, y = [], []

    for aa, angle in enumerate(angles):
        for idx, e in enumerate(emiters):

            x1, y1, x2, y2 = circlePoints[(angle, e)]
            if int(x1) != int(x2) and int(y1) != int(y2):
                a = (y2 - y1) / (x2 - x1)
                b = y1 - a * x1

                x1, y1 = 0, b
                x2, y2 = border, a * border + b

                x3, y3 = -b / a, 0
                x4, y4 = (border - b) / a, border

                points = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                points = list(filter(lambda x: 0 <= x[0] <= border and 0 <= x[1] <= border, points))
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
                print("#PRZYPAŁ")  # jeden przypadek dzielenia przez zero, lol
                addValue = 0
            else:
                addValue = sinogramData[aa][idx] / (len(pixels))

                newImageFlat = np.reshape(newImage, newshape=-1)
                newPixels = np.array(list(map(lambda x: x[1] + x[0] * image.shape[0], pixels)), dtype=np.uint32)
                newImageFlat[newPixels] += addValue
                newImage = np.reshape(newImageFlat, newshape=image.shape)

                countLinesFlat = np.reshape(countLines, newshape=-1)
                countLinesFlat[newPixels] += 1.0
                countLines = np.reshape(countLinesFlat, newshape=image.shape)

        x.append(aa)
        y.append(RMSE(image, newImage))

    if doCountLines:
        countLines[countLines == 0.0] = 1.0
        newImage /= countLines

    print('iks de')
    plt.gcf().clear()
    plt.close()

    plt.figure()
    plt.plot(x, y)
    plt.show()

    plt.figure()

    newImageRaw = newImage.copy()

    newImage = normalize2(newImage) * 255
    newImage = ndimage.gaussian_filter(newImage, sigma=0.5)
    #newImage = normalize(newImage, np.max(image))
    newImage = cutDownValues(newImage, cutDownValue=np.percentile(newImage, q=60))


    # Plot the original and the radon transformed image
    plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(ndimage.rotate(sinogramData, -90), cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(newImageRaw, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(newImage, cmap='gray')
    plt.xticks([]), plt.yticks([])
    # plt.show()
    plt.savefig("result.png")

    print('Finally: ' + str(RMSE(image, newImage)))
    return y[-1]


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import time
from penciltools import img2gray, gray2gradient, hist_match, yuv2rgb
from scipy import signal

time1 = time.time()


def pencil_draw(img_gray):
    G = gray2gradient(img_gray)

    # 8个方向的线
    length = int(min(G.shape[0], G.shape[1]) / 40)
    if length % 2:
        half_length = int((length + 1) / 2)
    else:
        half_length = int(length / 2)

    L = np.zeros([length, length, 8])
    line_width = 2
    for n in range(8):
        if n == 0 or n == 1 or n == 2 or n == 7:
            for x in range(length):
                y = half_length - int((x - half_length) * math.tan(math.pi / 8 * n))
                if y > 0 and y < length:
                    for t in range(line_width):
                        L[y - t, x, n] = 1
            if n == 0 or n == 1 or n == 2:
                L[:, :, n + 4] = np.rot90(L[:, :, n])
    L[:, :, 3] = np.rot90(L[:, :, 7], 3);

    Gi = np.zeros([G.shape[0], G.shape[1], 8])
    for n in range(8):
        Gi[:, :, n] = signal.convolve2d(G, L[:, :, n], mode='same')
        print('Gi:' + str(1 / 8 * (n + 1) * 100) + "%")

    G_index = np.argmax(Gi, 2)
    Ci = np.zeros([G.shape[0], G.shape[1], 8])
    for n in range(8):
        Ci[:, :, n] = G * (G_index == n)

    Si = np.zeros([G.shape[0], G.shape[1], 8])
    for n in range(8):
        Si[:, :, n] = signal.convolve2d(Ci[:, :, n], L[:, :, n], mode='same')
        print('Si:' + str(1 / 8 * (n + 1) * 100) + "%")

    S = np.sum(Si, 2)
    S_normal = (S - np.amin(S)) / (np.amax(S) - np.min(S))

    from scipy.misc import imresize

    texture = img2gray('texture/pencil.jpg')
    H = imresize(texture, [img_gray.shape[0], img_gray.shape[1]])

    test = mpimg.imread('sketch/test.png')
    J = hist_match(img_gray, test)

    J_normal = (J - np.amin(J)) / (np.amax(J) - np.min(J))
    H_normal = (H - np.amin(H)) / (np.amax(H) - np.min(H))
    beta = np.log(J_normal + 0.0001) / np.log(H_normal + 0.0001)

    beta_int = np.round(beta, decimals=2)
    T_normal = H_normal ** (0.7 * beta_int ** 0.6)

    R = T_normal * (1 - S_normal)
    time2 = time.time()
    print("用时" + str(time2 - time1) + "秒")
    return R


def pencil_draw_color(image_address):
    from scipy.misc import imread
    yuv = imread(image_address, mode='YCbCr')  # 主页这里的imread是从scipy中的imread
    Ypencil = pencil_draw(yuv[:, :, 0])
    new_yuv = yuv
    new_yuv[:, :, 0] = Ypencil * 255
    R = yuv2rgb(new_yuv)
    return R


image = 'img/test3.jpg'
img_gray = img2gray(image)
R = pencil_draw(img_gray)
plt.subplot(1, 2, 1)
plt.imshow(R, cmap='gray')

R = pencil_draw_color(image)
plt.subplot(1, 2, 2)
plt.imshow(R)
plt.show()


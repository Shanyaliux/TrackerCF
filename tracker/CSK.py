import cv2
import numpy as np
import pylab

from utils import *


class CSK:
    def __init__(self, imgPath):
        self.imgPath = imgPath
        self.frameList = getImgList(self.imgPath)
        self.frameList.sort()

        # 根据论文参数
        self.padding = 1.0  # 目标周围的额外区域
        # spatial bandwidth (proportional to target)
        self.output_sigma_factor = 1 / float(16)
        self.sigma = 0.2  # 高斯核带宽
        self.lambda_value = 1e-2  # 正则化
        self.interpolation_factor = 0.075  # 适应的线性插值因子

    def cos_window(self, sz):
        """
        width, height = sz
        j = np.arange(0, width)
        i = np.arange(0, height)
        J, I = np.meshgrid(j, i)
        cos_window = np.sin(np.pi * J / width) * np.sin(np.pi * I / height)
        """

        # cos_window = np.hanning(int(sz[1]))[:, np.newaxis].dot(np.hanning(int(sz[0]))[np.newaxis, :])
        cos_window = np.outer(np.hanning(int(sz[1])), np.hanning(int(sz[0])))
        return cos_window

    def gaussian2d_labels(self, sz, sigma):
        w, h = sz
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        center_x, center_y = w / 2, h / 2
        dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2) / (sigma ** 2)
        labels = np.exp(-0.5 * dist)
        return labels

    def _dgk(self, x1, x2):
        c = np.fft.fftshift(self.ifft2(self.fft2(x1) * np.conj(self.fft2(x2))))
        d = np.dot(x1.flatten().conj(), x1.flatten()) + np.dot(x2.flatten().conj(), x2.flatten()) - 2 * c
        k = np.exp(-1 / self.sigma ** 2 * np.clip(d, a_min=0, a_max=None) / np.size(x1))
        return k

    def fft2(self, x):
        # return np.fft.fft(np.fft.fft(x, axis=1), axis=0).astype(np.complex64)
        return np.fft.fft2(x)

    def ifft2(self, x):
        # return np.fft.ifft(np.fft.ifft(x, axis=1), axis=0).astype(np.complex64)
        return np.fft.ifft2(x)

    def _training(self, x, y):
        k = self._dgk(x, x)
        alphaf = self.fft2(y) / (self.fft2(k) + self.lambda_value)
        return alphaf

    def _detection(self, alphaf, x, z):
        k = self._dgk(x, z)
        responses = np.real(self.ifft2(alphaf * self.fft2(k)))
        return responses

    def track(self):

        info = loadImgInfo(self.frameList[0], 'CSK')
        initImg, initGt, target_sz, center = info

        initImg = cv2.cvtColor(initImg, cv2.COLOR_BGR2GRAY)
        initImg = initImg.astype(np.float32)

        x, y, w, h = tuple(initGt)

        self.w, self.h = w, h
        self._window = getCosWindow((int(round(2 * w)), int(round(2 * h))))
        self.crop_size = (int(round(2 * w)), int(round(2 * h)))
        self.x = cv2.getRectSubPix(initImg, (int(round(2 * w)), int(round(2 * h))), center) / 255 - 0.5
        self.x = self.x * self._window
        s = np.sqrt(w * h) / 16
        self.y = self.gaussian2d_labels((int(round(2 * w)), int(round(2 * h))), s)
        self._init_response_center = np.unravel_index(np.argmax(self.y, axis=None), self.y.shape)
        self.alphaf = self._training(self.x, self.y)

        for idx in range(len(self.frameList)):
            img = cv2.imread(self.frameList[idx])
            current_frame = img
            if len(current_frame.shape) == 3:
                assert current_frame.shape[2] == 3
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            current_frame = current_frame.astype(np.float32)
            z = cv2.getRectSubPix(current_frame, (int(round(2 * self.w)), int(round(2 * self.h))),
                                  center) / 255 - 0.5
            z = z * self._window
            self.z = z
            responses = self._detection(self.alphaf, self.x, z)
            curr = np.unravel_index(np.argmax(responses, axis=None), responses.shape)
            dy = curr[0] - self._init_response_center[0]
            dx = curr[1] - self._init_response_center[1]
            x_c, y_c = center
            x_c -= dx
            y_c -= dy
            center = (x_c, y_c)
            new_x = cv2.getRectSubPix(current_frame, (2 * self.w, 2 * self.h), center) / 255 - 0.5
            new_x = new_x * self._window
            self.alphaf = self.interpolation_factor * self._training(new_x, self.y) + \
                          (1 - self.interpolation_factor) * self.alphaf
            self.x = self.interpolation_factor * new_x + (1 - self.interpolation_factor) * self.x
            rect = np.array([center[0] - self.w / 2, center[1] - self.h / 2, self.w, self.h]).astype(np.int64)
            # print(rect)
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)
            cv2.imshow('CSK', img)
            cv2.waitKey(20)


img_path = '../datasets/surfer'
tracker = CSK(img_path)
# tracker.track()
s = np.sqrt(200 * 150) / 16
y = tracker.gaussian2d_labels((int(round(2 * 200)), int(round(2 * 150))), s)

cv2.imshow("1", y)
cv2.waitKey(0)



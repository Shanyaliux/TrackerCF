import os

# 获取图片列表
import cv2
import numpy as np


def getImgList(imgPath):
    frameList = []
    for frame in os.listdir(imgPath):
        if os.path.splitext(frame)[1] == '.jpg':
            frameList.append(os.path.join(imgPath, frame))
    return frameList


def loadImgInfo(img, windowName='Tracker'):
    # 获取第一帧图像
    initImg = cv2.imread(img)
    # 获取跟踪框
    initGt = cv2.selectROI(windowName, initImg, False, False)
    initGt = np.array(initGt).astype(np.int64)
    # 获取跟踪框尺寸 height width
    targetSize = np.array([initGt[3], initGt[2]])
    # 获取跟踪框中心点坐标
    center = (initGt[0] + targetSize[1] / 2, initGt[1] + targetSize[0] / 2)

    ret = [initImg, initGt, targetSize, center]
    return ret


# 生成余弦窗口
def getCosWindow(sz):
    return np.outer(np.hanning(int(sz[1])), np.hanning(int(sz[0])))


# 归一化
def linearMapping(img):
    return (img - img.min()) / (img.max() - img.min())


# 高斯响应
def getGuassResponse(img, gt, sigma):
    # 获取图像的高宽
    height, width = img.shape
    # 获取网格
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    # 获取跟踪框的重点坐标
    centerX = gt[0] + 0.5 * gt[2]
    centerY = gt[1] + 0.5 * gt[3]
    # 计算距离
    distance = (np.square(xx - centerX) + np.square(yy - centerY)) / (2 * sigma)
    # 获取响应图
    response = np.exp(-distance)
    # 归一化
    response = linearMapping(response)

    return response[gt[1]: gt[1] + gt[3], gt[0]: gt[0] + gt[2]]

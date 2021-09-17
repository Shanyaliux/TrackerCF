from utils import *


class Mosse:
    def __init__(self, img_path):

        self.sigma = 100
        self.lr = 0.125
        self.num_pretrain = 128
        self.rotate = True
        self.img_path = img_path
        self.frameList = getImgList(self.img_path)
        self.frameList.sort()

    # 开始跟踪
    def track(self):
        global clipPos, pos

        info = loadImgInfo(self.frameList[0], 'Mosse')
        initImg, initGt, target_sz, center = info

        # 灰度处理
        initFrame = cv2.cvtColor(initImg, cv2.COLOR_BGR2GRAY)
        initFrame = initFrame.astype(np.float32)

        # 获取高斯响应
        g = getGuassResponse(initFrame, initGt, 100)

        # 从完整图像上抠出跟踪框
        fi = initFrame[initGt[1]: initGt[1] + initGt[3], initGt[0]: initGt[0] + initGt[2]]
        # 快速傅里叶变换
        G = np.fft.fft2(g)

        # 预处理
        Ai, Bi = self.preTraining(fi, G)

        # 开始跟踪
        for idx in range(len(self.frameList)):
            currentFrame = cv2.imread(self.frameList[idx])
            frameGray = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
            frameGray = frameGray.astype(np.float32)
            if idx == 0:
                Ai = self.lr * Ai
                Bi = self.lr * Bi
                pos = initGt.copy()
                clipPos = np.array([pos[0], pos[1], pos[0] + pos[2], pos[1] + pos[3]]).astype(np.int64)
            else:
                Hi = Ai / Bi  # 公式10
                fi = frameGray[clipPos[1]:clipPos[3], clipPos[0]:clipPos[2]]
                # 预处理
                fi = self.preProcess(cv2.resize(fi, (initGt[2], initGt[3])))

                Gi = Hi * np.fft.fft2(fi)   # 公式 1
                gi = linearMapping(np.fft.ifft2(Gi))

                maxValue = np.max(gi)
                maxPos = np.where(gi == maxValue)
                dy = int(np.mean(maxPos[0]) - gi.shape[0] / 2)
                dx = int(np.mean(maxPos[1]) - gi.shape[1] / 2)

                # 更新坐标
                pos[0] = pos[0] + dx
                pos[1] = pos[1] + dy

                clipPos[0] = np.clip(pos[0], 0, currentFrame.shape[1])
                clipPos[1] = np.clip(pos[1], 0, currentFrame.shape[0])
                clipPos[2] = np.clip(pos[0] + pos[2], 0, currentFrame.shape[1])
                clipPos[3] = np.clip(pos[1] + pos[3], 0, currentFrame.shape[0])
                clipPos = clipPos.astype(np.int64)

                fi = frameGray[clipPos[1]:clipPos[3], clipPos[0]:clipPos[2]]
                fi = self.preProcess(cv2.resize(fi, (initGt[2], initGt[3])))

                # 公式 11 和 12
                Ai = self.lr * (G * np.conjugate(np.fft.fft2(fi))) + (1 - self.lr) * Ai
                Bi = self.lr * (np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))) + (1 - self.lr) * Bi

            cv2.rectangle(currentFrame, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (255, 0, 0), 2)
            cv2.imshow('Mosse', currentFrame)
            cv2.waitKey(20)

    # 预训练
    def preTraining(self, initFrame, G):
        # 获取跟踪框大小
        height, width = G.shape
        # 统一尺寸
        fi = cv2.resize(initFrame, (width, height))
        # 图像预处理
        fi = self.preProcess(fi)
        # 计算 Ai Bi      公式 8 和 10
        Ai = G * np.conjugate(np.fft.fft2(fi))
        Bi = np.fft.fft2(initFrame) * np.conjugate(np.fft.fft2(initFrame))
        # 随机变换
        for _ in range(self.num_pretrain):
            if self.rotate:
                fi = self.preProcess(self.randomWarp(initFrame))
            else:
                fi = self.preProcess(initFrame)
            Ai = Ai + G * np.conjugate(np.fft.fft2(fi))
            Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))
        return Ai, Bi

    # 图像预处理
    def preProcess(self, img):
        # 获取图像的尺寸
        height, width = img.shape
        # 引入log函数
        img = np.log(img + 1)
        img = (img - np.mean(img)) / (np.std(img) + 1e-5)
        # 引入余弦窗
        window = np.outer(np.hanning(height), np.hanning(width))
        img = img * window
        return img

    # 图像随机仿射变换
    def randomWarp(self, img):
        a = -180 / 16
        b = 180 / 16
        r = a + (b - a) * np.random.uniform()
        # rotate the image...
        matrix_rot = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), r, 1)
        img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
        img_rot = img_rot.astype(np.float32) / 255
        return img_rot

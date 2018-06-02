import numpy as np
from scipy.misc import *
import os


def load_image(filename):
    try:
        im = imread(filename)
    except:
        im = None
    return im


class water_svd(object):
    def __init__(self, A, W, alpha):
        self.A = A
        self.W = W
        self.alpha = alpha

    def rebuild_img(self, u, sigma, v):
        m = len(u)
        n = len(v)
        a = np.zeros((m, n))

        k = 0
        while k < sigma.shape[0]:
            uk = u[:, k].reshape(m, 1)
            vk = v[k].reshape(1, n)
            a += sigma[k] * np.dot(uk, vk)

            k += 1
        a[a < 0] = 0
        a[a > 255] = 255

        return np.rint(a).astype("uint8")

    def addImage(self, k):
        U, Sig, V = np.linalg.svd(A[:, :, k])
        L = np.zeros([Sig.shape[0], Sig.shape[0]])
        for i in range(Sig.shape[0]):  # i<768
            L[i][i] += Sig[i]
            for j in range(Sig.shape[0]):  # j<768
                if i < W.shape[0] and j < W.shape[1]:  # i<300 j<800
                    L[i][j] += self.alpha * W[i][j][k]  # a是水印强度,加上水印
        U_1, Sig_1, V_1 = np.linalg.svd(L)
        return self.rebuild_img(U, Sig_1, V)

    def reduction_image(self, k, P):
        U, Sig, V = np.linalg.svd(self.A[:, :, k])
        L = np.zeros([Sig.shape[0], Sig.shape[0]])
        for i in range(Sig.shape[0]):
            L[i][i] += Sig[i]
            for j in range(Sig.shape[0]):
                if i < self.W.shape[0] and j < self.W.shape[1]:
                    L[i][j] += self.alpha * self.W[i][j][k]
        U_1, Sig_1, V_1 = np.linalg.svd(L)
        p = P[:, :, k]
        U_p, Sig_p, V_p = np.linalg.svd(p)
        F = self.rebuild_img(U_1, Sig_p, V_1)
        for i in range(Sig.shape[0]):
            F[i][i] -= Sig[i]
        return F / self.alpha

    def encode(self, target_path):
        R = self.addImage(0)
        G = self.addImage(1)
        B = self.addImage(2)
        I = np.stack((R, G, B), 2)
        imsave(target_path, I)

    def decode(self, target_path, P):
        R = self.reduction_image(0, P)
        G = self.reduction_image(1, P)
        B = self.reduction_image(2, P)
        I = np.stack((R, G, B), 2)
        imsave(target_path, I)


base_path = os.path.join("", "../picture/")
origin_path = os.path.join(base_path, "original_pic.jpg")
water_path = os.path.join(base_path, "water_marking.jpg")
if __name__ == "__main__":
    A = load_image(origin_path)
    W = load_image(water_path)
    alpha = [0.2, 0.5, 1.0]

    # for i in range(len(alpha)):
    #     print("This is " + str(i + 1) + "th experiment.")
    #     test = water_svd(A, W, alpha[i])
    #     print("start to encode...")
    #     save_path = os.path.join(base_path, "encode_pic_" + str(alpha[i]) + ".jpg")
    #     test.encode(save_path)
    #     print("encode finished...")
    #
    #     P = load_image(save_path)
    #     decode_path = os.path.join(base_path, "decode_mark_" + str(alpha[i]) + ".jpg")
    #     print("start to decode...")
    #     test.decode(decode_path, P)
    #     print("decode finished...")

    for i in range(len(alpha)):
        print("This is " + str(i + 1) + "th experiment.")
        W = np.zeros((W.shape[0], W.shape[1], W.shape[2]))
        test = water_svd(A, W, alpha[2])
        print("start to decode original image...")
        save_path = os.path.join(base_path, "origin_decode_" + str(alpha[i]) + ".jpg")
        P = load_image(origin_path)
        test.decode(save_path, P)
        print("original image decode finished...")

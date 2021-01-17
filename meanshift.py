import numpy as np
from scipy.stats import multivariate_normal
STOP_THRESHOLD = 1e-4
CLUSTER_THRESHOLD = 1e-1


def lengthan(one, two):
    dist = np.array(one) - np.array(two)
    return np.linalg.norm(dist)


def gaussianKernelFunction(juli, bandwidth):
    left = (bandwidth * np.sqrt(2 * np.pi))
    right = -0.5 * ((juli / bandwidth)) ** 2
    # multivariate_normal(mean=1, cov=bandwidth).pdf(1+juli)
    return (1 / left) * np.exp(right)


class MeanShift(object):
    def __init__(self, kernel=gaussianKernelFunction):
        self.kernel = kernel

    def running_dots(self, onedata, data, bandwidth):
        N,D=data.shape
        dot=np.zeros(D)
        # print(D)
        # dotX = 0.0
        # dotY = 0.0
        scale = 0.0
        for eachdot in data:
            dist = lengthan(onedata, eachdot)
            quanzhong = self.kernel(dist, bandwidth)
            # for i in range(D):
            dot=dot+ eachdot * quanzhong
            # dotX = dotX + eachdot[0] * quanzhong
            # dotY = dotY + eachdot[1] * quanzhong
            scale = scale + quanzhong
        # for i in range(D):
        dot=dot/scale
        # dotX = dotX / scale
        # dotY = dotY / scale
        return dot

    def productLabelCluster(self, points):
        labelindexs = []
        labelindexx = 0
        ms_centers = []

        for num, dot in enumerate(points):
            if (len(labelindexs) == 0):
                labelindexs.append(labelindexx)
                ms_centers.append(dot)
                labelindexx = labelindexx + 1
            else:
                for center in ms_centers:
                    dist = lengthan(dot, center)
                    if (dist < CLUSTER_THRESHOLD):
                        labelindexs.append(ms_centers.index(center))
                if (len(labelindexs) < num + 1):
                    labelindexs.append(labelindexx)
                    ms_centers.append(dot)
                    labelindexx = labelindexx + 1
        return ms_centers, labelindexs

    def product_result(self, data, bandwidth):
        piaoyi = [True] * data.shape[0]
        running_dots = np.array(data)
        while (1):
            biggest_long = 0
            for each in range(0, len(running_dots)):
                if not piaoyi[each]:
                    continue
                starting_moving = running_dots[each].copy()
                running_dots[each] = self.running_dots(running_dots[each], data, bandwidth)
                dist = lengthan(running_dots[each], starting_moving)
                biggest_long = max(biggest_long, dist)
                piaoyi[each] = dist > STOP_THRESHOLD

            if (biggest_long < STOP_THRESHOLD):
                break
        ms_centers, lables = self.productLabelCluster(running_dots.tolist())
        return running_dots, lables, ms_centers


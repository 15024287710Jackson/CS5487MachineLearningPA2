import pandas as pd
import numpy as np
import kmeans
import random
import sklearn.cluster as sk
import EMGMM
import meanshift
import matplotlib.pyplot as plt
from sklearn import mixture
import sklearn.cluster as ms
from sklearn.cluster import estimate_bandwidth

def read_data(file):
    dataSet=[]
    fileIn = open(file)
    for line in fileIn.readlines():
        temp = []
        lineArr = line.strip().split('\t')
        for each_sample in lineArr:
            temp.append(float(each_sample))
        dataSet.append(temp)
    fileIn.close()
    return np.array(dataSet)

def auccary(pidect_labels,ture_labels):
    long=len(pidect_labels)
    pidct=0;
    for i in range(long):
        if(int(pidect_labels[i]) ==int(ture_labels[i])):
            pidct=pidct+1
    acc=float(pidct)/float(long)
    return acc

def colors(n):
  col = []
  for i in range(n):
    col.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
  return col
# Press the green button in the gutter to run the script.

def main_kmeans_function(dataAx,k_class,pic_n,label,data_Ay):
    datamatA = np.mat(dataAx)
    centers, cluster_label_accessment = kmeans.kmeans(datamatA, k_class)
    clusterAssment = np.array(cluster_label_accessment)
    kmeans_label = clusterAssment[:, 0]
    print(centers)
    color = colors(np.unique(kmeans_label).size)
    plt.figure(pic_n)
    plt.title('kmean_data'+label)
    for i in range(len(dataAx)):
        plt.scatter(dataAx[i, 0], dataAx[i, 1], color=color[int(kmeans_label[i])])
    # estimator = sk.KMeans(n_clusters=4)
    # res = estimator.fit_predict(dataAx)
    # center = estimator.cluster_centers_
    # labels = estimator.labels_
    # print(center)
    # plt.title('sklearn_kmeans_data'+label)
    # plt.figure(pic_n+1)
    # for i in range(len(dataAx)):
    #     plt.scatter(dataAx[i, 0], dataAx[i, 1], color=color[labels[i]])
    # plt.figure(pic_n+2)
    # print(data_Ay.shape)
    # plt.title('true_data'+label)
    # for i in range(len(data_Ay[0])):
    #     plt.scatter(dataAx[i, 0], dataAx[i, 1], color=color[int(data_Ay[0, i]) - 1])

def GMMEM_function(dataAx,k_class,times,pic_n,label):
    datamatA = np.mat(dataAx)
    mu, cov, alpha = EMGMM.GMM_EM(datamatA, k_class, times)
    gamma = EMGMM.calculationEstep(datamatA, mu, cov, alpha)
    gmmem_labels = gamma.argmax(axis=1).flatten().tolist()[0]
    plt.figure(pic_n)
    plt.scatter(dataAx[:, 0], dataAx[:, 1], c=gmmem_labels, s=40, cmap='viridis')
    plt.title("GMM Clustering By EM Algorithm data"+label)
    # gmm = mixture.GaussianMixture(n_components=k_class).fit(datamatA)
    # labels = gmm.predict(datamatA)
    # plt.figure(pic_n+1)
    # plt.title("GaussianMixture"+label)
    # plt.scatter(dataAx[:, 0], dataAx[:, 1], c=labels, s=40, cmap='viridis')

def meanshift_function(dataAx,bandwidth,quantile,pic_n,label):
    mean_shifter = meanshift.MeanShift()
    __, mean_shift_result, mscenters = mean_shifter.product_result(dataAx, bandwidth=bandwidth)
    plt.figure(pic_n)
    plt.title("meanshift data"+label)
    color = colors(np.unique(mean_shift_result).size)
    for i in range(len(mean_shift_result)):
        plt.scatter(dataAx[i, 0], dataAx[i, 1], color=color[mean_shift_result[i]])
    # bandwidth1 = estimate_bandwidth(dataAx, quantile=quantile)
    # clf = ms.MeanShift(bandwidth=bandwidth1, n_jobs=-1)
    # clf.fit(dataAx)
    # labels = clf.labels_
    # plt.figure(pic_n+1)
    # color2 = colors(np.unique(labels).size)
    # plt.title("sklearn.cluster data"+label)
    # for i in range(len(dataAx)):
    #     plt.scatter(dataAx[i][0], dataAx[i][1], color=color2[labels[i]])

if __name__ == '__main__':
    #  sample data
       data_Ax=read_data("./cluster_data_text/cluster_data_dataA_X.txt")
       dataAx=data_Ax.transpose()
       data_Bx=read_data("./cluster_data_text/cluster_data_dataB_X.txt")
       dataBx=data_Bx.transpose()
       data_Cx=read_data("./cluster_data_text/cluster_data_dataC_X.txt")
       dataCx=data_Cx.transpose()
    #  labels
       data_Ay = read_data("./cluster_data_text/cluster_data_dataA_Y.txt")
       data_By = read_data("./cluster_data_text/cluster_data_dataB_Y.txt")
       data_Cy = read_data("./cluster_data_text/cluster_data_dataC_Y.txt")
       k_class=4
       # kmeans- implement
       main_kmeans_function(dataAx, k_class, 1, 'A',data_Ay)
       main_kmeans_function(dataBx, k_class, 2, 'B', data_By)
       main_kmeans_function(dataCx, k_class, 3, 'C', data_Cy)
       # GMM-EM
       times = 150
       GMMEM_function(dataAx, k_class, times, 4, 'A')
       GMMEM_function(dataBx, k_class, times, 5, 'B')
       GMMEM_function(dataCx, k_class, times, 6, 'C')
       # meanshift
       meanshift_function(dataAx, 0.11, 0.3, 7, 'A')
       meanshift_function(dataBx, 0.11, 0.2, 8, 'B')
       meanshift_function(dataCx, 0.1, 0.4, 9, 'C')
       plt.show()

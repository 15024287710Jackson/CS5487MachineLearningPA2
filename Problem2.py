import pa2
import numpy as np
import pylab as pl
from PIL import Image
import random
import kmeans
import sklearn.cluster as sk
import EMGMM
from sklearn import mixture
import meanshift
from sklearn.cluster import estimate_bandwidth
import sklearn.cluster as ms
def colors(n):
  col = []
  for i in range(n):
    col.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
  return col

def main_kmeans_function(dataAx,k_class):
    datamatA = np.mat(dataAx)
    centers, cluster_label_accessment = kmeans.kmeans(datamatA, k_class)
    clusterAssment = np.array(cluster_label_accessment)
    kmeans_label = clusterAssment[:, 0]
    # estimator = sk.KMeans(n_clusters=k_class)
    # res = estimator.fit_predict(dataAx)
    # labels = estimator.labels_
    return kmeans_label#,labels

def GMMEM_function(dataAx,k_class,times):
    datamatA = np.mat(dataAx)
    mu, cov, alpha = EMGMM.GMM_EM(datamatA, k_class, times)
    gamma = EMGMM.calculationEstep(datamatA, mu, cov, alpha)
    gmmem_labels = gamma.argmax(axis=1).flatten().tolist()[0]
    # gmm = mixture.GaussianMixture(n_components=k_class).fit(datamatA)
    # labels = gmm.predict(datamatA)
    return np.array(gmmem_labels)#,np.array(labels)

def pic_prod(Y,pic,pic_label):
    Y = Y + 1  # Use matlab 1-index labeling
    ##
    pl.figure(pic)
    pl.subplot(1, 3, 1)
    pl.title('original')
    pl.imshow(img)
    # make segmentation image from labels
    segm = pa2.labels2seg(Y, L)
    pl.subplot(1, 3, 2)
    pl.imshow(segm)

    # color the segmentation image
    csegm = pa2.colorsegms(segm, img)
    pl.title(pic_label)
    pl.subplot(1, 3, 3)
    pl.imshow(csegm)

def meanshift_function(dataAx,bandwidth,quantile):
    # mean_shifter = meanshift.MeanShift()
    # __, mean_shift_result, mscenters = mean_shifter.product_result(dataAx, bandwidth=bandwidth)

    bandwidth = estimate_bandwidth(dataAx, quantile=quantile)
    # print(bandwidth)
    clf = ms.MeanShift(bandwidth=bandwidth, n_jobs=-1)
    clf.fit(dataAx)
    labels = clf.labels_
    return np.array(labels)#np.array(mean_shift_result),


if __name__ == '__main__':
    import scipy.cluster.vq as vq
    ## load and show image
    num_pic='12003'
    img = Image.open('images/'+num_pic+'.jpg')
    ## extract features from image (step size = 7)
    X, L = pa2.getfeatures(img, 7)
    ## Call kmeans function in scipy.  You need to write this yourself!
    # C, Y = vq.kmeans2(vq.whiten(X.T), 2, iter=1000, minit='random')
    k_class=3

    kmeans_Y= main_kmeans_function(vq.whiten(X.T), k_class)
    pic_prod(kmeans_Y, 1,num_pic+'_kmeans')
    # pic_prod(kmeans_Y1, 2,num_pic+'_sklearn_kmeans')

    times = 150
    GMMEM_Y=GMMEM_function(vq.whiten(X.T), k_class, times)
    pic_prod(GMMEM_Y, 2,num_pic+'_GMMEM')
    # pic_prod(GMMEM_Y1, 4,num_pic+'_mixture_GaussianMixture')

    meanshift_Y1=meanshift_function(vq.whiten(X.T), 0.5, 0.1)
    # pic_prod(meanshift_Y, 5,num_pic+'_meanshift')
    pic_prod(meanshift_Y1, 3,num_pic+'_ms.MeanShift')
    # problem 2b
    data=vq.whiten(X.T)
    # print(data[0])
    lamda=0.5
    data[:, 2] *= lamda
    data[:, 3] *= lamda
    # print(data[0])
    kmeans_Y_lamda= main_kmeans_function(data, k_class)
    pic_prod(kmeans_Y_lamda, 4,num_pic+'_kmeans_lamda='+str(lamda))
    # pic_prod(kmeans_Y1, 8,num_pic+'_sklearn_kmeans_lamda='+str(lamda))
    data_h = vq.whiten(X.T)
    h_c=1.5
    h_p=0.5
    data_h[:, 0] *= h_c
    data_h[:, 1] *= h_c
    data_h[:, 2] *= h_p
    data_h[:, 3] *= h_p
    meanshift_Y1 = meanshift_function(data_h, 0.5, 0.15)
    pic_prod(meanshift_Y1, 5, num_pic + '_ms.MeanShift_h_c_h_p')
    pl.show()
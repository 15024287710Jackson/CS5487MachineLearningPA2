from numpy import *
def lenddist(pointA,pointB):
    return sqrt(sum(power(pointA-pointB,2)))

def define_center(data,num):
    count_data,lie = data.shape
    centers = zeros((num,lie))
    for i in range(num):
        locations= int(random.uniform(0,count_data))
        centers[i,:]=data[locations,:]
    return centers


def kmeans(data, k):
    count_data = data.shape[0]
    cluster_label = mat(zeros((count_data, 2)))
    centers = define_center(data, k)
    judege = True
    while judege:
        judege = False
        for i in range(count_data):
            shortloc = 0
            shortdis = 100000.0
            for j in range(k):
                distance = lenddist(centers[j, :], data[i, :])
                if distance < shortdis:
                    shortloc = j
                    shortdis = distance
            if cluster_label[i, 0] != shortloc:
                cluster_label[i, :] = shortloc, shortdis ** 2
                judege = True
        for each in range(k):
            eachtypepoints = data[nonzero(cluster_label[:, 0].A == each)[0]]
            centers[each, :] = mean(eachtypepoints, axis=0)
    return centers, cluster_label
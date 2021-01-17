import numpy as np
from scipy.stats import multivariate_normal

def each_probability(dataset, kthmode_mu, kthmode_conv):
    guassian = multivariate_normal(mean=kthmode_mu, cov=kthmode_conv)
    return guassian.pdf(dataset)

def calculationEstep(dataset, mu, cov, alpha):
    num_sample = dataset.shape[0]
    num_model = alpha.shape[0]
    assert num_sample > 1, "please input num_sample which is more than one sample!"
    assert num_model > 1, "please input num_model which is more than one gaussian model!"
    gamma = np.mat(np.zeros((num_sample, num_model)))

    precent = np.zeros((num_sample, num_model))
    for k in range(num_model):
        precent[:, k] = each_probability(dataset, mu[k], cov[k])
    precent = np.mat(precent)

    for each in range(num_model):
        gamma[:, each] = alpha[each] * precent[:, each]
    for every in range(num_sample):
        gamma[every, :] /= np.sum(gamma[every, :])
    return gamma

def preprocessing(dataset):
    num_sample=dataset.shape[1]
    for i in range(num_sample):
        max_value = dataset[:, i].max()
        min_value = dataset[:, i].min()
        wide_range=max_value - min_value
        dataset[:, i] = (dataset[:, i] - min_value) / wide_range
    print("Data scaled.")
    return dataset

def define_original_params(shape, num_model):
    row, column = shape
    init_mu = np.random.rand(num_model, column)
    init_cov = np.array([np.eye(column)] * num_model)
    each_weight=[1.0 / num_model]
    init_alpha = np.array(each_weight * num_model)
    print("Original Parameters.")
    print("original_mu:", init_mu, "original_cov:", init_cov, "original_alpha:", init_alpha, sep="\n")
    return init_mu, init_cov, init_alpha

def Mstep(data, gamma):
    row, column = data.shape
    num_model = gamma.shape[1]

    max_mu = np.zeros((num_model, column))
    max_conv = []
    max_alpha = np.zeros(num_model)

    for every in range(num_model):
        kthmodel = np.sum(gamma[:, every])
        max_mu[every, :] = np.sum(np.multiply(data, gamma[:, every]), axis=0) / kthmodel
        result = (data - max_mu[every]).transpose() * np.multiply((data - max_mu[every]), gamma[:, every]) / kthmodel
        max_conv.append(result)
        max_alpha[every] = kthmodel / row
    max_conv = np.array(max_conv)
    return max_mu, max_conv, max_alpha

def GMM_EM(data, num_model, itera):
    data = preprocessing(data)
    look_like=data.shape
    mu, cov, alpha = define_original_params(look_like, num_model)
    for times in range(itera):
        gamma = calculationEstep(data, mu, cov, alpha)
        mu, cov, alpha = Mstep(data, gamma)
    print("{sep} Result {sep}".format(sep="-" * 20))
    print("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha


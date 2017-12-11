import matplotlib.pyplot as plt
import pandas
import numpy as np
from pomegranate import *
from math import exp, cos, sin
import scipy.stats as ss# Note: there are Gaussian mixture models in scikit.learn and pomegranate

## Example  -> see 2DGuassian in the lecture 10 files -> those covariance matrices are the same as below

d1 = MultivariateGaussianDistribution([0,0], [[3.23539123, 1.30736366], [1.30736366, 1.76460877]])
d2 = MultivariateGaussianDistribution([10,10], [[4.14954339,-2.41414444],[-2.41414444,2.85045661]])
d3 = MultivariateGaussianDistribution([-10,-10], [[17.67926173 , -8.48921224],[-8.48921224, 5.32073827]])
model = GeneralMixtureModel([d1, d2, d3])
sample_data = np.array(model.sample(1000))
[x, y] = sample_data.transpose()
plt.scatter(x, y)
plt.show()

#Create a some random Covariance matrices with seperatew means (to seperate them out visually)
def get_rand_mgd_list(num_distributions):
    dists = []
    x_mean = 0
    y_mean = 0

    for i in range(num_distributions):
        var = np.random.randint(2000) / 100
        angle = np.random.randint(1000) / 100
        V = np.array([[1.0, 0.0], [0.0, var]])
        R = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
        covariance_mat = np.dot(np.dot(R, V), np.transpose(R))
        dists.append(MultivariateGaussianDistribution([x_mean, y_mean], covariance_mat))
        x_mean += 5  #not sure whether we should randomize the mean?????
        y_mean += 5
    return dists

#creat the model, sample som data and predict which distribution its from (not necessary for the assignment but helpful to understand)
model = GeneralMixtureModel(get_rand_mgd_list(3))
sample_data_2 = np.array(model.sample(1000))
labels = model.predict(sample_data_2)
[x_0, y_0] = sample_data_2[labels == 0].transpose()
[x_1, y_1] = sample_data_2[labels == 1].transpose()
[x_2, y_2] = sample_data_2[labels == 2].transpose()
plt.scatter(x_0, y_0, color='red')
plt.scatter(x_1, y_1, color='blue')
plt.scatter(x_2, y_2, color='green')
plt.show()


model_2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=3, X=sample_data_2)
labels_2 = model_2.predict(sample_data_2)
[x_0_2, y_0_2] = sample_data_2[labels_2 == 0].transpose()
[x_1_2, y_1_2] = sample_data_2[labels_2 == 1].transpose()
[x_2_2, y_2_2] = sample_data_2[labels_2 == 2].transpose()
plt.scatter(x_0_2, y_0_2, color='red')
plt.scatter(x_1_2, y_1_2, color='blue')
plt.scatter(x_2_2, y_2_2, color='green')
plt.show()

print(model_2.predict_proba(sample_data_2))


# def plot2DGaussian(var_ratio, angle):
#     mean = [0, 0]
#     V = np.array([[1.0, 0.0], [0.0, var_ratio]])  # diagonal covariance
#     R = np.array([[cos(angle), -sin(angle)],[sin(angle), cos(angle)]])
#     cov = np.dot(np.dot(R,V),np.transpose(R))
#     print("V")
#     print(V)
#     print("R")
#     print(R)
#     print("cov")
#     print(cov)
#     print("mean")
#     print(mean)
#     x, y = np.random.multivariate_normal(mean, cov, 5000).T
#     plt.plot(x, y, 'x', alpha=0.4, marker='.')
#     plt.axis('equal')
#     plt.show()
#
# plot2DGaussian(1.0,0.0) #Circular
# plot2DGaussian(4.0,0.0) #Vertical Cigar
# plot2DGaussian(0.1,0.0) #Horizontal Cigar
# plot2DGaussian(4.0,2.1) #Diagonal (increasing L->R) cigar
# plot2DGaussian(6.0,7.2) #Diagonal (decreasing L->R) cigar
# plot2DGaussian(22.0,1.1) #Diagonal (decreasing L->R) cigar -Much Narrower

# def mix(x):
#     return (2.0/3.0)*ss.norm.pdf(x,loc=-1.0,scale=0.2*c)+(1.0/3.0)*ss.norm.pdf(x,loc=0,scale=0.25*c)
#
# def resp(x):
#     return ((2.0/3.0)*ss.norm.pdf(x,loc=-1.0,scale=0.2*c))/ mix(x)
#
# x = np.arange(-2., 1., 0.01)
#
# # for c in [2.0,1.5,1.0,0.75,0.5,0.25,0.125,0.0675]:
# #     plt.plot(x,mix(x),label='Mixture')
# #     plt.plot(x,resp(x),label='Responsibility')
# #     plt.legend(loc='upper right')
# #     plt.show()
# for c in [2.0]:
#     plt.plot(x,mix(x),label='Mixture')
#     plt.plot(x,resp(x),label='Responsibility')
#     plt.legend(loc='upper right')
#     plt.show()


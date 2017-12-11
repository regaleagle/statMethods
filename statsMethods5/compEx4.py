import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas
from matplotlib.mlab import griddata
import numpy as np
from pomegranate import *
from math import exp, cos, sin
import scipy.stats as ss# Note: there are Gaussian mixture models in scikit.learn and pomegranate


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

def plot_predictions(model, arbitrary_component_num):
    #get original generated predictions
    sample_data = np.array(model.sample(1000))
    labels = model.predict(sample_data)
    [x_0, y_0] = sample_data[labels == 0].transpose()
    [x_1, y_1] = sample_data[labels == 1].transpose()
    [x_2, y_2] = sample_data[labels == 2].transpose()

    #create model from sample data with arbitrary component number for some reason.
    # Should we use random initialiser as above then fit instead?
    model_2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=arbitrary_component_num, X=sample_data)

    [x_it, y_it] = sample_data.transpose()

    #Create grid for contour
    x_range = np.arange(x_it.min(), x_it.max(), 0.1)
    y_range = np.arange(y_it.min(), y_it.max(), 0.1)
    xgrid, ygrid = np.meshgrid(x_range, y_range)
    zgrid = np.zeros_like(xgrid)
    for i in range(len(xgrid)):
        for j in range(len(xgrid[i])):
            zgrid[i][j] = model_2.probability(np.array([xgrid[i][j], ygrid[i][j]]))

    plt.contour(xgrid, ygrid, zgrid, 10)
    plt.scatter(x_0, y_0, color='red', alpha=0.2)
    plt.scatter(x_1, y_1, color='blue', alpha=0.2)
    plt.scatter(x_2, y_2, color='green', alpha=0.2)
    plt.show()


## Example  -> see 2DGuassian in the lecture 10 files -> those covariance matrices are the same as below
d1 = MultivariateGaussianDistribution([0,0], [[3.23539123, 1.30736366], [1.30736366, 1.76460877]])
d2 = MultivariateGaussianDistribution([10,10], [[4.14954339,-2.41414444],[-2.41414444,2.85045661]])
d3 = MultivariateGaussianDistribution([-10,-10], [[17.67926173 , -8.48921224],[-8.48921224, 5.32073827]])
model = GeneralMixtureModel([d1, d2, d3])
sample_data_1 = np.array(model.sample(1000))
[x, y] = sample_data_1.transpose()
plt.scatter(x, y)

delta = 0.025
x = np.arange(-20.0, 20.0, delta)
y = np.arange(-20.0, 20.0, delta)
X, Y = np.meshgrid(x, y)

#possible plotting solution based on parameters:
Z0 = mlab.bivariate_normal(X, Y, 3.23539123, 1.76460877, 0.0, 0.0, 1.30736366)

Z1 = mlab.bivariate_normal(X, Y, 4.14954339, 2.85045661, 10, 10, -2.41414444)

Z2 = mlab.bivariate_normal(X, Y, 17.67926173, 5.32073827, -10, -10, -8.48921224)

CS0 = plt.contour(X, Y, Z0)
CS1 = plt.contour(X, Y, Z1)
CS2 = plt.contour(X, Y, Z2)
plt.clabel(CS0, inline=1, fontsize=10)
plt.clabel(CS1, inline=1, fontsize=10)
plt.clabel(CS2, inline=1, fontsize=10)
plt.title('Simplest default with labels')
plt.show()



#creat the model, sample som data and predict which distribution its from (not necessary for the assignment but helpful to understand)
model = GeneralMixtureModel(get_rand_mgd_list(3))

# probably what we're supposed to do?
plot_predictions(model, 2)
plot_predictions(model, 3)
plot_predictions(model, 4)




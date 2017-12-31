import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas
from matplotlib.mlab import griddata
import matplotlib.cm as cm
import numpy as np
from pomegranate import *
from math import exp, cos, sin, log
import scipy.stats as ss# Note: there are Gaussian mixture models in scikit.learn and pomegranate


#Create a some random Covariance matrices with seperate means (to seperate them out visually)
def get_rand_mgd_list(num_distributions):
    dists = []
    x_mean = 0
    y_mean = 0

    for i in range(num_distributions):
        var = np.random.randint(3000) / 100
        angle = np.random.randint(1000) / 100
        V = np.array([[1.0, 0.0], [0.0, var]])
        R = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
        covariance_mat = np.dot(np.dot(R, V), np.transpose(R))
        dists.append(MultivariateGaussianDistribution([x_mean, y_mean], covariance_mat))
        x_mean += 5  #not sure whether we should randomize the mean?????
        y_mean += 5
    return dists

def plot_predictions(model, arbitrary_component_num, sample_info):
    #get original generated predictions
    sample_data = np.array(sample_info)
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

    plt.contour(xgrid, ygrid, zgrid, 15, linewidths=1)
    plt.scatter(x_0, y_0, color='red', alpha=0.2)
    plt.scatter(x_1, y_1, color='blue', alpha=0.2)
    plt.scatter(x_2, y_2, color='green', alpha=0.2)
    plt.title(f'With {arbitrary_component_num} components')
    plt.show()


# SOURCES: https://www.youtube.com/watch?v=qMTuMa86NzU, https://www.youtube.com/watch?v=QQJHsKfNqG8

def gen_responsibilities(samples, distributions):
    responsibilities = []
    for vector in samples:
        gaussian_probs = [dist.probability(vector) for dist in distributions]
        normalization_factor = sum(gaussian_probs)
        responsibilities.append([prob / normalization_factor for prob in gaussian_probs])
    
    return np.array(responsibilities)

# NOTE: Look up 'Shannon entropy'
def determine_entropy(responsibilities):
    entropies = []
    for resps in responsibilities:
        entropy = -sum([r*log(r, 3) for r in resps])
        entropies.append(entropy)
    return entropies

## Example  -> see 2DGuassian in the lecture 10 files -> those covariance matrices are the same as below
# d1 = MultivariateGaussianDistribution([0,0], [[3.23539123, 1.30736366], [1.30736366, 1.76460877]])
# d2 = MultivariateGaussianDistribution([10,10], [[4.14954339,-2.41414444],[-2.41414444,2.85045661]])
# d3 = MultivariateGaussianDistribution([-10,-10], [[17.67926173 , -8.48921224],[-8.48921224, 5.32073827]])
# model = GeneralMixtureModel([d1, d2, d3])
# sample_data_1 = np.array(model.sample(1000))
# [x, y] = sample_data_1.transpose()
# plt.scatter(x, y)
#
# delta = 0.025
# x = np.arange(-20.0, 20.0, delta)
# y = np.arange(-20.0, 20.0, delta)
# X, Y = np.meshgrid(x, y)
#
# #possible plotting solution based on parameters:
# Z0 = mlab.bivariate_normal(X, Y, 3.23539123, 1.76460877, 0.0, 0.0, 1.30736366)
#
# Z1 = mlab.bivariate_normal(X, Y, 4.14954339, 2.85045661, 10, 10, -2.41414444)
#
# Z2 = mlab.bivariate_normal(X, Y, 17.67926173, 5.32073827, -10, -10, -8.48921224)
#
# CS0 = plt.contour(X, Y, Z0)
# CS1 = plt.contour(X, Y, Z1)
# CS2 = plt.contour(X, Y, Z2)
# plt.clabel(CS0, inline=1, fontsize=10)
# plt.clabel(CS1, inline=1, fontsize=10)
# plt.clabel(CS2, inline=1, fontsize=10)
# plt.title('Simplest default with labels')
# plt.show()



#creat the model, sample som data and predict which distribution its from (not necessary for the assignment but helpful to understand)
distributions = get_rand_mgd_list(3)
model = GeneralMixtureModel(distributions)
sample_points = model.sample(1000)
[x_val, y_val] = np.array(sample_points).T

# # probably what we're supposed to do?
plot_predictions(model, 2, sample_points)
plot_predictions(model, 3, sample_points)
plot_predictions(model, 4, sample_points)

# Question 2:
resps = gen_responsibilities(sample_points, distributions)
entropies = determine_entropy(resps)

#My guess for plotting entropies
hot_map = plt.get_cmap('hot')

plt.scatter(x_val, y_val, c=[x for x in entropies], cmap=hot_map)
plt.colorbar()
plt.show()


#Q3

def get_initial_state_params(sequence, num_states):
    means_stds = []
    lower = sequence.min()
    for i in range(1,num_states+1):
        perc = (100/num_states) * i
        upper = np.percentile(sequence, perc)
        sample_set = sequence[(sequence > lower) & (sequence < upper)]
        sample = np.random.choice(sample_set, len(sample_set)//4, replace=False)
        lower = upper
        means_stds.append((np.around(sample.mean(), 3), np.around(sample.std(), 3)))
    return means_stds

def get_random_transitions(num_states):
    mat = np.empty(shape=(num_states,num_states))
    for i in range(0,num_states):
        not_valid = True
        while not_valid:
            not_valid = False
            mat[i] = (np.random.dirichlet(np.ones(num_states),size=1))
            mat[i] = np.around(mat[i], 2)
            offset = 1 - np.sum(mat[i])
            if offset != 0:
                mat[i][np.random.randint(num_states)] += offset
            for elem in mat[i]:
                if elem <= 0:
                    not_valid = True
    return mat

def create_hmm(means_and_stds, transition_matrix):
    states = []
    count = 0
    for mean, variance in means_and_stds:
        state = State(NormalDistribution(mean, variance), name="S"+str(count))
        states.append(state)
        count += 1
    model = HiddenMarkovModel()
    # model.start = states[0]
    # model.end = states[len(states)-1]
    model.add_states(states)

    for state in states:
        model.add_transition(model.start, state, 0.33)

    for row in range(len(transition_matrix)):
        for col in range(len(transition_matrix[row])):
            model.add_transition(states[row], states[col], transition_matrix[row][col])
    model.bake()
    np.random.seed(int(time.time() // 1000))
    return model

PROBE_ID_COL = 0
CHROMOSOMAL_POS_COL = 1
FIRST_PATIENT_COL = 2

hyb_data = pandas.read_csv('hyb.txt', sep='\t', header=None)
chromo_pos = np.array(hyb_data[CHROMOSOMAL_POS_COL])
patient_sequences = hyb_data.as_matrix()[2:].transpose()
patient_indices_list = [6,9,10]

def generate_hmms(data, patient_indices):
    model_list = []
    number_of_states = 3
    for i in patient_indices:
        means_and_stds = get_initial_state_params(data[i], number_of_states)
        trans_mat = get_random_transitions(number_of_states)
        hmm_model = create_hmm(means_and_stds, trans_mat)
        model_list.append(hmm_model)
        # number_of_states += 1  #Uncomment to have multiple states in HMMs
    return model_list

hmm_models = generate_hmms(patient_sequences, patient_indices_list)

##THIS IS WHAT IT SEEMS LIKE HE IS ASKING FOR BUT IT FEELS LIKE COMPLETE NONSENSE
hmm_gmm_model = GeneralMixtureModel(hmm_models)

hmm_gmm_model.fit(patient_sequences)

predictions = hmm_gmm_model.predict(patient_sequences)
plt.scatter(range(len(predictions)), predictions)
plt.show()

count_uniq = 0
for i in range(len(predictions)):
    if count_uniq >= len(hmm_gmm_model.distributions):
        break
    if predictions[i] == count_uniq:
        # hmm_gmm_model.distributions[predictions[i]]
        print("hmm: ", predictions[i])
        print(hmm_gmm_model.distributions[predictions[i]].dense_transition_matrix())
        logp, path = hmm_gmm_model.distributions[predictions[i]].viterbi(patient_sequences[i])
        path_num = np.array([int(str(x)[1]) for x in path])
        plt.scatter(range(len(patient_sequences[i])), patient_sequences[i], s=2, c=path_num[1:])
        plt.show()
        count_uniq += 1

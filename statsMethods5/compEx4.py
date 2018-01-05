import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas
from matplotlib.mlab import griddata
import matplotlib.cm as cm
import numpy as np
from pomegranate import *
from math import exp, cos, sin, log
import scipy.stats as ss# Note: there are Gaussian mixture models in scikit.learn and pomegranate

# Q1
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



#create the model, sample som data and predict which distribution its from (not necessary for the assignment but helpful to understand)
distributions = get_rand_mgd_list(3)
model = GeneralMixtureModel(distributions)
sample_points = model.sample(1000)
[x_val, y_val] = np.array(sample_points).T

plot_predictions(model, 2, sample_points)
plot_predictions(model, 3, sample_points)
plot_predictions(model, 4, sample_points)

# Q2:
resps = gen_responsibilities(sample_points, distributions)
entropies = determine_entropy(resps)

hot_map = plt.get_cmap('hot')

plt.scatter(x_val, y_val, c=[x for x in entropies], cmap=hot_map)
plt.colorbar()
plt.show()


# Q3 & Q4

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
    model.add_states(states)

    for state in states:
        model.add_transition(model.start, state, 0.33)
        #model.add_transition(state, model.end, 0.33)

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
patient_sequences = hyb_data.as_matrix().transpose()[2:]



def generate_unique_hmms(data, list_of_state_size, number_of_hmms):
    model_list = []
    patient_indices = np.random.randint(len(data), size=number_of_hmms)
    random_transitions = [get_random_transitions(list_of_state_size[i]) for i in range(len(patient_indices))]

    for i in range(len(patient_indices)):
        means_and_stds = get_initial_state_params(data[patient_indices[i]], list_of_state_size[i])
        trans_mat = random_transitions[i]
        hmm_model = create_hmm(means_and_stds, trans_mat)
        hmm_model.fit([data[patient_indices[i]]])
        model_list.append(hmm_model)
    return model_list


# Creates HMMs starting with SAME transitions between all the HMM's and same number of states
def generate_duplicate_hmms(data, number_of_states):
    model_list = []
    patient_indices = np.random.randint(len(data), size=number_of_states)
    random_transition = get_random_transitions(number_of_states)

    for i in patient_indices:
        initial_state_param = get_initial_state_params(data[i], number_of_states)    
        hmm_model = create_hmm(initial_state_param, random_transition)
        model_list.append(hmm_model)
        hmm_model.fit([data[i]])
        # number_of_states += 1  #Uncomment to have multiple states in HMMs
    return model_list

def plot_prediction_distribution(predictions):
    plt.subplot(211)
    plt.scatter(range(len(predictions)), predictions)
    plt.subplot(212)
    plt.hist(predictions)
    plt.show()

def plot_gmm_predictions(predictions, patient_sequences, gmm_model):
    predicted = []
    to_predict = set(predictions)
    predict_lists = {}
    for i in to_predict:
        predict_lists[i] = []
        for j in range(len(patient_sequences)):
            if predictions[j] == i:
                predict_lists[i].append(patient_sequences[j])
        random_index = np.random.randint(len(predict_lists[i]))
        logp, path = gmm_model.distributions[i].viterbi(predict_lists[i][random_index])
        path_num = np.array([int(str(x)[1]) for x in path])
        plt.title(f'Number of states: {len(gmm_model.distributions[i].states)-2}')
        plt.scatter(range(len(predict_lists[i][random_index])), predict_lists[i][random_index], s=2, c=path_num[1:])
        plt.show()

hmm_models = generate_unique_hmms(patient_sequences, [3, 3, 3, 4], 4)
hmm_gmm_model = GeneralMixtureModel(hmm_models)
hmm_gmm_model.fit(patient_sequences)

predictions = hmm_gmm_model.predict(patient_sequences)

plot_prediction_distribution(predictions)
plot_gmm_predictions(predictions, patient_sequences, hmm_gmm_model)

dup_hmm_models = generate_duplicate_hmms(patient_sequences, 3)
dup_hmm_gmm_model = GeneralMixtureModel(dup_hmm_models)
dup_hmm_gmm_model.fit(patient_sequences)

dup_predictions = dup_hmm_gmm_model.predict(patient_sequences)

plot_prediction_distribution(dup_predictions)
plot_gmm_predictions(dup_predictions, patient_sequences, dup_hmm_gmm_model)

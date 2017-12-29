import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas
from matplotlib.mlab import griddata
import matplotlib.cm as cm
import numpy as np
from pomegranate import *
from math import exp, cos, sin, log
import scipy.stats as ss# Note: there are Gaussian mixture models in scikit.learn and pomegranate

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
patient_data = np.array(hyb_data)[:,2:].transpose() # Transpose so we can access by patient instead of by chromosomal pos

def generate_hmms(data, number_of_hmms, number_of_states):
    model_list = []

    # All 'random' stuff has to be done here as the call to create_hmm below resets the random seed
    seed_patients = [np.random.randint(len(data)) for i in range(number_of_hmms)] 
    random_transition_matrices = [get_random_transitions(number_of_states) for i in range(number_of_hmms)]

    for i in range(number_of_hmms):
        random_mean_and_stds = get_initial_state_params(data[seed_patients[i]], number_of_states)
        random_transition_matrix = random_transition_matrices[i]
        hm_model = create_hmm(random_mean_and_stds, random_transition_matrix)
        # hm_model.fit(data)
        model_list.append(hm_model)
    
    return model_list

# NOTE: Look up 'Shannon entropy'
def determine_entropy(responsibilities):
    entropies = []
    for resps in responsibilities:
        entropy = -sum([r*log(r, 3) for r in resps])
        entropies.append(entropy)
    return entropies

hm_models = generate_hmms(patient_data, 3, 4)

# # Just to make sure we're on the right track
# for m in hm_models:
#     plt.plot(m.sample(500))
#     plt.show()

mixture_model = GeneralMixtureModel(hm_models)
mixture_model.fit(patient_data)
sample_data = patient_data[2] # mixture_model.sample() is timing out for me, so using a random patient as the sample data instead

responsibilities = mixture_model.predict_proba(sample_data)
entropies = determine_entropy(responsibilities)

plt.subplot(211)
plt.scatter(chromo_pos, sample_data, c=mixture_model.predict(sample_data))
plt.subplot(212)
plt.scatter(chromo_pos, sample_data, c=entropies, cmap=plt.get_cmap('hot'))
plt.colorbar()
plt.show()
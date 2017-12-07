import matplotlib.pyplot as plt
import pandas
import numpy as np
from math import *
from pomegranate import *
from textwrap import wrap
import time

# Constants

PROBE_ID_COL = 0
CHROMOSOMAL_POS_COL = 1
FIRST_PATIENT_COL = 2

hyb_data = pandas.read_csv('hyb.txt', sep='\t', header=None)
chromo_pos = np.array(hyb_data[CHROMOSOMAL_POS_COL])
first_patient_ratio = np.array(hyb_data[FIRST_PATIENT_COL])


## Task 1
# RATIOS = CHROMOSOMAL_POS_COL. There is absolutely no way to know this other than to ask someone who already knows.
# Are we sure about this? to me it looks like the ratios are FIRST_PATIENT_COL at the chromosomal position - James
#
plt.scatter(chromo_pos, first_patient_ratio, s=2)
plt.title('Best Guess Segmentation')
plt.xlabel('position')
plt.ylabel('ratio')
plt.ylim(-0.5,1) #Looks better but maybe bad? - James

#Highlighting possible segments
plt.axvspan(0, 7200000, color='red', alpha=0.5)
plt.axvspan(7200000, 32000000, color='blue', alpha=0.5)
plt.axvspan(78000000, 120000000, color='red', alpha=0.5)
plt.axvspan(167000000, 182000000, color='blue', alpha=0.5)
plt.axvspan(182000000, 213000000, color='red', alpha=0.5)
plt.axvspan(248000000, 270000000, color='red', alpha=0.5)
plt.axvspan(276000000, 307000000, color='blue', alpha=0.5)
plt.show()

plt.hist(first_patient_ratio, range=(-1, 1), bins=30)
#Looks a little nicer with the range, no idea if it is a good idea -James
plt.show()

#Does anyone know what he means by thresholds? Trimming the long tail data maybe?
#Threshold could be the 99th percentile, that's what I've added below, -James

## Task 2
# INPUT:
#   means_and_variances:
#       List of states with their Guassian PDF means and variances: [(mean, variance), (mean, variance) ....]
#
#   transitions:
#       Transition matrix such that, for the transition between the state 0 and state 1 will be located at row 0 col 1
#           [0.1    _0.5_      0.6]
#           [0.3     0.1       0.3]
#           [0.9     0.4       0.2]
#
#       NOTE: If col value is -1, we assume it means model.end (i.e. the transition between the final state and the end of the model)
#
#
# RETURNS:
#   Trained model and the generated states (in order to enable path calculations etc)

def create_hmm(means_and_stds, transition_matrix):
    states = []
    count = 0;
    for mean, variance in means_and_stds:
        state = State(NormalDistribution(mean, variance), name="S"+str(count))
        states.append(state)
        count += 1
    model = HiddenMarkovModel()
    model.add_states(states)

    for state in states:
        model.add_transition(model.start, state, 0.33)

        # Not sure whether we should define state end and
        # whether or not we should randomize entry or just define 1 start state?

    for row in range(len(transition_matrix)):
        for col in range(len(transition_matrix[row])):
            model.add_transition(states[row], states[col], transition_matrix[row][col])
    model.bake()
    np.random.seed(int(time.time() // 1000))
    return model

def create_seg_plot(sequence, t, path, initial_params):
    plt.scatter(t, sequence, s=2)
    title = ""
    count = 0
    for param in initial_params:
        title += "\n" + f'S{count}: M {param[0]}, Std {param[1]}'
        count += 1
    plt.title(title)
    plt.xlabel('position')
    plt.ylabel('ratio')
    plt.ylim(-0.5, 1)
    start = 0
    if path[0][1].name == "None-start":
        path.pop(0)
    current_state = path[0][1].name
    for i in range(len(t)):
        if path[i][1].name != current_state:
            if current_state == 'S0':
                plt.axvspan(start, t[i], color='red', alpha=0.5)
            elif current_state == 'S2':
                plt.axvspan(start, t[i], color='blue', alpha=0.5)
            start = t[i]
            current_state = path[i][1].name
    plt.show()

def train_model(model, sequences):
    model.fit([sequences]) # Baum-welch training by default
    logp, path = model.viterbi(sequences) # Viterby segment
    return path

#Task 4

# a) Looking at the histogram, one way we could randomize initial
# parameters (instead of taking the uniformly segmented mean and std)
# is to take a small sample from different quantile ranges of the data set
# and use the the mean and variance of those samples as our initial Gaussian PDFs.
# For the transitions we will simply choose a random distribution from 0-1 for
# each state's transition row.

# b)Using the default Baum-Welch parameters 1e8 max iterations, and 1e-9
# stop threshold, we usually get results with less than 100 iterations,
# from this we determined that the stop threshold on improvement was more
# relevant and the default limit set was fine

# c)While the problem specifically is looking for 3 states, duplication, deletion or norm
# the data distribution in the initial histogram suggest shows some limited
# extreme values on the right tail. This will skew the segmentation somewhat.
# We could trim that data to clean it or we could introduce a 4th state that
# highlights the extreme data. We've chosen to go with the former to keep our methods generic

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
        mat[i] = (np.random.dirichlet(np.ones(num_states),size=1))
        mat[i] = np.around(mat[i], 2)
        offset = 1 - np.sum(mat[i])
        if offset != 0:
            mat[i][np.random.randint(num_states)] += offset
    return mat

# def get_random_transitions(num_states):
#     mat = np.empty(shape=(num_states,num_states))
#     for i in range(0,num_states):
#         rowtotal = 1
#         for j in range(0,num_states):
#             if j == num_states - 1:
#                 mat[i][j] = rowtotal
#             else:
#                 mat[i][j] = numpy.random.randint(rowtotal * 100)/100
#                 rowtotal -= mat[i][j]
#     print(mat)
#     return mat

#Clean
threshold = np.percentile(first_patient_ratio, 99)
first_patient_ratio_clean = first_patient_ratio[(first_patient_ratio < threshold)]
chromo_pos_clean = chromo_pos[(first_patient_ratio < threshold)]

#Random Parameters
number_of_states = 3
means_and_stds = get_initial_state_params(first_patient_ratio_clean, number_of_states)
trans_mat = get_random_transitions(number_of_states)

#Build Model
hmm_model = create_hmm(means_and_stds, trans_mat)

# Train and Display
path = train_model(hmm_model, first_patient_ratio_clean)
create_seg_plot(first_patient_ratio_clean, chromo_pos_clean, path, means_and_stds)

##Task 5


def test_model_prediction(m_and_stds, trans_m):
    #Random Parameters Test

    #Build Model Test
    new_model = create_hmm(m_and_stds, trans_m)
    sample_test_1 = new_model.sample(length=1000, path=True)

    #Train and Display Test
    path = train_model(new_model, np.array(sample_test_1[0]))

    # print(", ".join([p.name for p in sample_test_1[1]]))
    # print(", ".join([p[1].name for p in path]))
    total = len(sample_test_1[0])
    mismatches = 0
    for samp, pred in zip(sample_test_1[1], path):
        if samp.name != pred[1].name:
            mismatches += 1
    print("accuracy:")
    print(100*((total-mismatches)/total))
    return (100*((total-mismatches)/total))

test_model_prediction(means_and_stds, trans_mat)

# Task 6
# The posterior plot generated is overall flat with several major dips. This is likely due to some elements of the sequence being very random/unlikely to be next -
# regardless of which state emits them. If a state has very low in-bound transition probabilities from other states, or if a state has very low out-bound probabilities towards other
# states, the 'dip' will be observed. Overall this method does relatively well, however it fails to consider outlying/extreme possibilities.

def compute_posteriors(sequence, model):
    forward = model.forward(sequence)
    backward = model.backward(sequence)
    all_posteriors = []

    for i in range(len(forward)):
        probs = forward[i]
        posteriors = [None for _ in probs]

        for j in range(len(probs)):
            posteriors[j] = forward[i][j] + backward[i][j] # As these are log probabilities we will add them as opposed to multiplying them as if they were normal probabilities.

        all_posteriors.append(posteriors)

    return all_posteriors

posteriors = compute_posteriors(first_patient_ratio_clean, hmm_model)
plt.plot([max(x) for x in posteriors])
plt.show()


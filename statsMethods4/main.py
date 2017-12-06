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
plt.scatter(chromo_pos, first_patient_ratio)
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

def create_hmm(means_and_variances, transition_matrix):
    states = []
    count = 0;
    for mean, variance in means_and_variances:
        state = State(NormalDistribution(mean, sqrt(variance)), name="S"+str(count))
        states.append(state)
        count += 1
    model = HiddenMarkovModel()
    model.add_states(states)

    # TODO: Transitions, HOW CAN WE KNOW!??!
    model.add_transition(model.start, states[0], 0.33) # GUESS: I suppose we know for sure that the first state is reached for certain.
    model.add_transition(model.start, states[1], 0.33)
    model.add_transition(model.start, states[2], 0.33)

    for row in range(len(transition_matrix)):
        for col in range(len(transition_matrix[row])):
            model.add_transition(states[row], states[col], transition_matrix[row][col])
    model.bake()
    #np.random.seed(int(time.time() // 1000))
    return model

## Task 3
# def seg_print(sequences, states):
#     for i,s in enumerate(sequences):
#         print("Data {}: {}".format(i, "".join(str(o) for o in s)))
#         print("State: {}".format(states[i][0])) # As our states dont haeve names, use the index

def create_seg_plot(sequence, t, path, initial_params):
    plt.plot(t, sequence)
    plt.title("Init_Params" + "\n".join(wrap(str(initial_params), 50)))
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
    model.fit([sequences], max_iterations=10000) # Baum-welch training by default
    logp, path = model.viterbi(sequences) # Viterby segment
    return path

#Task 4

# a) Looking at the histogram, one way we could randomize initial
# parameters is to take a small sample from different quantile ranges of the data set
# and use the the mean and variance of those samples as our initial Gaussian PDFs.
# For the transitions we could will simply choose a random distribution from 0-1 for
# each state (although we will enforce a min value of 0.2 based on trial and error)

# b)After some trial an error we found that sometimes the algorithm can get close to
# it's max iterations so we limit it at 10000 which seems to give us good results
#  --Not really sure what he wants here - James

# c)While the problem specifically is looking for 3 states, duplication, deletion or norm
# the data distribution in the initial histogram suggest shows some limited
# extreme values on the right tail. This will skew the segmentation somewhat.
# We could trim that data to clean it or we could introduce a 4th state that
# highlights the extreme data. We've chosen to go with the former to keep our methods generic

def get_initial_state_params(sequence, num_states):
    means_variances = []
    lower = sequence.min()
    for i in range(1,num_states+1):
        perc = (100/num_states) * i
        upper = np.percentile(sequence, perc)
        set = sequence[(sequence > lower) & (sequence < upper)]
        sample = np.random.choice(set, len(set)//4, replace=False)
        lower = upper
        means_variances.append((sample.mean(), sample.std()))
    return means_variances

def get_random_transitions(num_states):
    mat = np.empty(shape=(num_states,num_states))
    for i in range(0,num_states):
        not_valid = True
        while not_valid:
            not_valid = False
            mat[i] = (np.random.dirichlet(np.ones(num_states),size=1))
            for elem in mat[i]:
                if elem <= 0.2:
                    not_valid = True
    return mat

#Clean
threshold = np.percentile(first_patient_ratio, 99)
first_patient_ratio_clean = first_patient_ratio[(first_patient_ratio < threshold)]
chromo_pos_clean = chromo_pos[(first_patient_ratio < threshold)]

#Random Parameters
number_of_states = 3
means_and_variances = get_initial_state_params(first_patient_ratio_clean, number_of_states)
trans_mat = get_random_transitions(number_of_states)

#Build Model
hmm_model = create_hmm(means_and_variances,trans_mat)

# Train and Display
path = train_model(hmm_model, first_patient_ratio_clean)
create_seg_plot(first_patient_ratio_clean, chromo_pos_clean, path, means_and_variances)

##Task 5


def test_model_prediction(model):
    # number_of_states_test = 3
    sample_test_1 = model.sample(length=700, path=True)

    #Random Parameters Test
    number_of_states = 3
    means_and_variances = get_initial_state_params(np.array(sample_test_1[0]), number_of_states)
    trans_mat = get_random_transitions(number_of_states)

    #Build Model Test
    new_model = create_hmm(means_and_variances,trans_mat)

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

test_model_prediction(hmm_model)

import matplotlib.pyplot as plt
import pandas
import numpy as np
from pomegranate import *
from math import exp
import time

# Constants

PROBE_ID_COL = 0
CHROMOSOMAL_POS_COL = 1
FIRST_PATIENT_COL = 2

hyb_data = pandas.read_csv('hyb.txt', sep='\t', header=None)
chromo_pos = np.array(hyb_data[CHROMOSOMAL_POS_COL])
first_patient_ratio = np.array(hyb_data[FIRST_PATIENT_COL])


## Task 1

plt.scatter(chromo_pos, first_patient_ratio, s=2)
plt.title('Best Guess Segmentation')
plt.xlabel('position')
plt.ylabel('ratio')

#Highlighting possible segments
plt.axvspan(0, 7200000, color='red', alpha=0.5)
plt.axvspan(7200000, 32000000, color='blue', alpha=0.5)
plt.axvspan(78000000, 120000000, color='red', alpha=0.5)
plt.axvspan(167000000, 182000000, color='blue', alpha=0.5)
plt.axvspan(182000000, 213000000, color='red', alpha=0.5)
plt.axvspan(248000000, 270000000, color='red', alpha=0.5)
plt.axvspan(276000000, 307000000, color='blue', alpha=0.5)
plt.show()

plt.hist(first_patient_ratio, range=(-1, 1), bins=30) #Exluding the outliers
plt.show()

## Task 2

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

    for row in range(len(transition_matrix)):
        for col in range(len(transition_matrix[row])):
            model.add_transition(states[row], states[col], transition_matrix[row][col])
    model.bake()
    np.random.seed(int(time.time() // 1000))
    return model

def create_seg_plot(sequence, t, path, initial_params, trans_matrix):
    plt.scatter(t, sequence, s=2)
    info = ""
    count = 0
    for param in initial_params:
        info += "\n" + f'S{count}: M {param[0]}, Std {param[1]}'
        count += 1
    plt.title("Segmentation Prediction")
    plt.text(0, 0.8, "means & std:" + str(info), fontsize=8)
    plt.text(150000000, 0.8, "trans mat:\n" + str(trans_matrix), fontsize=8)
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
                    not_valid = True;
    return mat

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

#Train and Display
path = train_model(hmm_model, first_patient_ratio_clean)
create_seg_plot(first_patient_ratio_clean, chromo_pos_clean, path, means_and_stds, trans_mat)

##Task 5


def test_model_prediction(m_and_stds, trans_m):
    #Random Parameters Test

    #Build Model Test
    new_model = create_hmm(m_and_stds, trans_m)
    sample_test_1 = new_model.sample(length=1000, path=True)
    real_path = [int(real.name[1]) for real in sample_test_1[1][1:]]# obviously won't work for more than 9 states...

    #Train and Display Test
    train_path = train_model(new_model, np.array(sample_test_1[0]))
    predict_path = [int(pred_path[1].name[1]) for pred_path in train_path[1:]]# obviously won't work for more than 9 states...

    # print(", ".join([p[1].name for p in path]))
    total = len(real_path)
    mismatches = 0
    real_states = []
    predictions = []
    x_val = []
    count_pos = 0
    for samp, pred in zip(real_path, predict_path):
        if samp != pred:
            mismatches += 1
            real_states.append(samp)
            predictions.append(pred)
            x_val.append(count_pos)
        count_pos += 1

    accuracy = 100 * ((total - mismatches) / total)
    info = ""
    count = 0
    for param in means_and_stds:
        info += "\n" + f'S{count}: M {param[0]}, Std {param[1]}'
        count += 1
    for x in x_val:
        plt.axvline(x=x, color='grey', alpha=0.2, linewidth=0.5)
    plt.scatter(x_val, real_states, color='blue', s=4)
    plt.scatter(x_val, predictions, color='red', s=4)
    plt.title(f'Mismatches - total accuracy: {accuracy}')
    plt.xlabel('position in sequence')
    plt.ylabel('state of path/prediction')
    plt.xlim(0,total)
    plt.text(total//2, 1.5, "means & std:" + str(info), fontsize=8)
    plt.text(total//4, 1.5, "trans mat:\n" + str(trans_m), fontsize=8)
    plt.show()
    print("accuracy:")
    print(accuracy)
    return accuracy

#test if randomly generated parameters used above are useful/correct.
test_model_prediction(means_and_stds, trans_mat)

# Task 6
# The posterior plot generated is overall flat with several major dips.
# This is likely due to some elements of the sequence being very random/unlikely to be next -
# regardless of which state emits them. If a state has very low in-bound
# transition probabilities from other states, or if a state has very low out-bound probabilities towards other
# states, the 'dip' will be observed. Overall this method does relatively
# well, however it fails to consider outlying/extreme possibilities.

def compute_posteriors(sequence, model):
    forward = model.forward(sequence)
    backward = model.backward(sequence)
    base_probability = model.log_probability(sequence)
    all_posteriors = []

    for i in range(len(forward)):
        probs = forward[i]
        posteriors = [None for _ in probs]

        for j in range(len(probs)):
            posteriors[j] = (forward[i][j] + backward[i][j]) - base_probability # As these are log probabilities we will add and subtract them as opposed to multiplying them as if they were normal probabilities.

        all_posteriors.append(posteriors)

    return all_posteriors


posteriors = compute_posteriors(first_patient_ratio_clean, hmm_model)
plt.title("Posteriors")
plt.plot([exp(max(x)) for x in posteriors])
plt.show()

##Task 7

def boxplots_for_emissions(model, sequence):
    emissions = model.forward_backward(sequence)[1]
    transp_emiss_array = np.array(emissions).transpose()

    plt.boxplot([emission for emission in transp_emiss_array])
    plt.show()

boxplots_for_emissions(hmm_model, first_patient_ratio_clean)



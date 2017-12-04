import matplotlib.pyplot as plt
import pandas
from math import *
from pomegranate import *

# Constants

PROBE_ID_COL = 0
CHROMOSOMAL_POS_COL = 1
FIRST_PATIENT_COL = 2

hyb_data = pandas.read_csv('hyb.txt', sep='\t', header=None)

## Task 1
# RATIOS = CHROMOSOMAL_POS_COL. There is absolutely no way to know this other than to ask someone who already knows.
plt.plot(hyb_data[CHROMOSOMAL_POS_COL], hyb_data[FIRST_PATIENT_COL])
plt.show()

plt.hist(hyb_data[CHROMOSOMAL_POS_COL])
plt.show()

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
    
    for mean, variance in means_and_variances:
        state = State(NormalDistribution(mean, sqrt(variance)))
        states.append(state)
    
    model = HiddenMarkovModel()
    model.add_states(states)

    # TODO: Transitions, HOW CAN WE KNOW!??!
    model.add_transition(model.start, states[0], 1.0) # GUESS: I suppose we know for sure that the first state is reached for certain.
    
    for row in range(len(transition_matrix)):
        for col in range(len(transition_matrix[row])):
            model.add_transition(states[row], model.end if row == -1 else states[col], transition_matrix[row][col])
    
    model.bake()
    return model

## Task 3
def seg_print(sequences, states, sequence_length):
    for i,s in enumerate(sequences):
        print("Data {}: {}".format(i, "".join(str(o) for o in s)))
        print("State: {}".format(states[i][0])) # As our states dont haeve names, use the index

def train_model(model, sequences):
    model.fit(sequences) # Baum-welch training

    segment = model.viterbi(sequences) # Viterby segment
    seg_print(sequences, segment[1], len(sequences[0]))

hmm_model = create_hmm([(0, 1), (1, 1), (2, 1)],   [[0.2,0.3,0.6],
                                                    [0.1,0.8,0.2],
                                                    [0.9,0.6,0.8]])

train_model(hmm_model, [[0, 1, 1, 0, 2, 1], [1, 2, 1, 1, 1, 1], [2, 2, 2, 1, 1, 1]])
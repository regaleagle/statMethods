import matplotlib.pyplot as plt
import pandas
import numpy as np
from math import *
from pomegranate import *

# Constants

PROBE_ID_COL = 0
CHROMOSOMAL_POS_COL = 1
FIRST_PATIENT_COL = 2

hyb_data = pandas.read_csv('hyb.txt', sep='\t', header=None)

## Task 1
# RATIOS = CHROMOSOMAL_POS_COL. There is absolutely no way to know this other than to ask someone who already knows.
# Are we sure about this? to me it looks like the ratios are FIRST_PATIENT_COL at the chromosomal position - James
#
plt.plot(hyb_data[CHROMOSOMAL_POS_COL], hyb_data[FIRST_PATIENT_COL])
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

plt.hist(hyb_data[FIRST_PATIENT_COL], range=(-1,1), bins=30)
#Looks a little nicer with the range, no idea if it is a good idea -James
plt.show()

#Does anyone know what he means by thresholds? -James

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
    model.add_transition(model.start, states[0], 1.0) # GUESS: I suppose we know for sure that the first state is reached for certain.
    
    for row in range(len(transition_matrix)):
        for col in range(len(transition_matrix[row])):
            model.add_transition(states[row], model.end if row == -1 else states[col], transition_matrix[row][col])
    
    model.bake()
    return model

## Task 3
def seg_print(sequences, states):
    for i,s in enumerate(sequences):
        print("Data {}: {}".format(i, "".join(str(o) for o in s)))
        print("State: {}".format(states[i][0])) # As our states dont haeve names, use the index

def create_seg_plot(sequence, t, path):
    plt.plot(t, sequence)
    plt.xlabel('position')
    plt.ylabel('ratio')
    plt.ylim(-0.5, 1)
    start = 0
    if path[0] == "None - start":
        path.pop(0)
    current_state = path[0]
    for i in range(len(t)):
        if path[i][1].name != current_state:

            if path[i][1].name == 'S0':
                plt.axvspan(start, t[i], color='red', alpha=0.5)
            elif path[i][1].name == 'S2':
                plt.axvspan(start, t[i], color='blue', alpha=0.5)

            current_state = path[i][1].name
            start = t[i]
    plt.show()

def train_model(model, sequences):
    model.fit([sequences]) # Baum-welch training
    logp, path = model.viterbi(sequences) # Viterby segment
    return path

hmm_model = create_hmm([(-0.1, .05), (0.0, 0.05), (0.1, 0.05)],
                                                    [[0.3,0.3,0.3],
                                                    [0.3,0.3,0.3],
                                                    [0.3,0.3,0.3]])

path = train_model(hmm_model, hyb_data[FIRST_PATIENT_COL])
create_seg_plot(hyb_data[FIRST_PATIENT_COL], hyb_data[CHROMOSOMAL_POS_COL], path)
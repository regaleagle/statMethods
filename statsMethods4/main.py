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

test_model = create_hmm([(0, 1), (1, 1), (2, 1)],   [[0.2,0.3,0.6],
                                                    [0.1,0.8,0.2],
                                                    [0.9,0.6,0.8]])


test_model.plot()
print(exp(test_model.log_probability([1,2,3]))) # What did I calculate? How are 1,2,3 valid emissions? How does it KNOW!??!?! :'D
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
#       List of transitions between 2 states and the probability of the transition occurring: [(s1, s2, prob), (s1, s3, prob), (s2, s3, prob), (s1, s1, prob), .... ]
#       With s1, s2 .... sN being the _INDEX_ of the state in the means_and_variances list
#       NOTE: If sN value is -1, we assume it means model.end (i.e. the transition between the final state and the end of the model)
#
#
# RETURNS:
#   Trained model

def create_hmm(means_and_variances, transitions):
    states = []
    
    for mean, variance in means_and_variances:
        state = State(NormalDistribution(mean, sqrt(variance)))
        states.append(state)
    
    model = HiddenMarkovModel()
    model.add_states(states)

    # TODO: Transitions, HOW CAN WE KNOW!??!
    model.add_transition(model.start, states[0], 1.0) # GUESS: I suppose we know for sure that the first state is reached for certain.
    
    for s1, s2, prob in transitions:
        model.add_transition(states[s1], model.end if s2 == -1 else states[s2], prob)
    
    model.bake()
    return model

test_model = create_hmm([(0, 1), (1, 1), (2, 1)], [(0, 1, 0.3), (0, 2, 0.7), (0, 0, 0.1), 
                                                    (1, 1, 0.5), (1, 2, 0.6), (1, 0, 0.1),
                                                    (2, 2, 0.6), (2, 0, 0.01), (2, 1, 0.6),
                                                    (2, -1, 0.9)])

print(test_model)
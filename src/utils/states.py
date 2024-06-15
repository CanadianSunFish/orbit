import numpy as np
from src.utils import constants as c

def get_states(data, state_type):

    states = []

    if (type(state_type) == str):
        state_type = [state_type]

    for i in range(len(state_type)):

        if state_type[i] == 'velocity':
            if(len(np.shape(data))==1):
                states.append(np.linalg.norm(data[3:6], axis=0))
            else:
                states.append(np.linalg.norm(data[3:6, :], axis=0))


        if state_type[i] == 'position':
            if(len(np.shape(data))==1):
                pos_data = np.linalg.norm(data[0:3], axis=0)
                states.append(pos_data - c.rplanet)

            else:
                pos_data = np.linalg.norm(data[0:3, :], axis=0)
                states.append(pos_data - c.rplanet)

    return states
import numpy as np

###############
# This program calculates the time of flight for a given m/q ratio
# (note that the constants are calculated using values which can change when changing the VMI voltages or the laser setup)
###############

mq = 127

###############
# Formular: t(m/q) = t_0 + const * sqrt(m/q)
###############

###############
# time center of MCP and mass/charge values for VMI 6.04/3.99
###############

'''# S+
t_1 = 3747.0
mq_1 = 32.0

# N+
t_2 = 3240
mq_2 = 14'''

#-------------------------------------------------------

###############
# time center of MCP and mass/charge values for VMI 3.0/1.95
###############

# CS2+
t_1 = 5565
mq_1 = 76

# S+
t_2 = 4400
mq_2 = 32


const = (t_1 - t_2) / ( np.sqrt(mq_1) - np.sqrt(mq_2))
t_0 = t_1 - const * np.sqrt(mq_1)

calc_t = t_0 + const * np.sqrt(mq)

print("Arrival time for m/q =", mq, ": ", round(calc_t, 2), "ns")
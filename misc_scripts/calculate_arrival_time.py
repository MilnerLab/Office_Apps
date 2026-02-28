import numpy as np

###############
# This program calculates the time of flight for a given m/q ratio
# (note that the constants are calculated using values which can change when changing the VMI voltages or the laser setup)
###############

mq = 14

###############
# Formular: t(m/q) = t_0 + const * sqrt(m/q)
###############

###############
# time center of MCP and mass/charge values for VMI 6.04/3.99
###############

t_sulfur = 3747.0
mq_sulfur = 32.0

t_iodine = 5310.0
mq_iodine = 127.0
np.sqrt



const = (t_sulfur - t_iodine) / ( np.sqrt(mq_sulfur) - np.sqrt(mq_iodine))
t_0 = t_sulfur - const * np.sqrt(mq_sulfur)

calc_t = t_0 + const * np.sqrt(mq)

print("Arrival time for m/q =", mq, ": ", round(calc_t, 2), "ns")